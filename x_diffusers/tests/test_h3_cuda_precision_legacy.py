"""Prove the new opt-in branches leave the previous default core AST intact."""
import ast
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='608142655eb5766280625b5923f1018b00f616e5'
FLAGS={'h3_cuda_precision':'none', 'h3_parallel_mode':'sp', 'h3_dit_tp':1,
       'h3_tp_comm':'bf16', 'h3_cp_comm':'auto', '_h3_tp_cp':False, '_h3_parallel_config':None}
FLAGS.update({name:False for name in (
    'h3_cuda_local_input','h3_cuda_local_output','h3_cuda_precompute_conditioning',
    '_h3_cuda_local_input','_h3_cuda_local_output','_h3_precompute_conditioning')})
FLAGS.update(h3_cuda_preset=None,_h3_cuda_precision_key=None,_h3_shared_sequence_exchange=None)
FLAGS.update(h3_cuda_linear_tp=1,h3_cuda_conditioning_offload=False,
    h3_cuda_reference_offload=False,h3_cuda_refiner_ff_chunk_size=0,
    h3_cuda_input_refiner='none',h3_cuda_vision_sp=False,
    h3_cuda_vae_encoder_parallel=False,h3_cuda_audio_encoder_parallel=False,
    h3_cuda_audio_decoder_parallel='none')
FLAGS.update(h3_vae_backend='legacy',h3_vae_model_path=None,h3_vae_trust_remote_code=False)


def constant(test):
    if any(isinstance(node,(ast.Name,ast.Call,ast.Attribute,ast.Subscript)) for node in ast.walk(test)):
        return None
    return bool(eval(compile(ast.Expression(test),'<constant>','eval'),{'__builtins__':{}}))


class DisabledPrecision(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        # The new keyword-only startup switch is omitted at old call sites.
        kept=[(arg,default) for arg,default in zip(node.args.kwonlyargs,node.args.kw_defaults)
              if arg.arg!='bootstrap_environment']
        node.args.kwonlyargs=[arg for arg,_ in kept]
        node.args.kw_defaults=[default for _,default in kept]
        return self.generic_visit(node)

    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)==3
                and isinstance(node.args[1],ast.Constant) and node.args[1].value in FLAGS):
            return ast.copy_location(ast.Constant(FLAGS[node.args[1].value]),node)
        if (isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)==3
                and isinstance(node.args[1],ast.Constant) and node.args[1].value=='_h3_vae_parallel_manager'):
            return node.args[2]
        if (isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)==3
                and isinstance(node.args[0],ast.Constant) and node.args[0].value is None):
            return node.args[2]
        if isinstance(node.func,ast.Name) and node.func.id=='needs_resolution':
            return ast.copy_location(ast.Constant(False),node)
        node.keywords=[kw for kw in node.keywords if not (
            kw.arg=='pad_key' and isinstance(kw.value,ast.Constant) and kw.value.value=='pad')]
        # Selective component loading contributes no kwargs when precision is off.
        node.keywords=[kw for kw in node.keywords if not (
            kw.arg is None and isinstance(kw.value,ast.Dict) and not kw.value.keys)]
        # Public H3 naming changes the shared argument's help, not Wan logic.
        if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--vae_lightning':
            node.keywords=[kw for kw in node.keywords if kw.arg!='help']
        return node

    def visit_Name(self,node):
        if node.id=='component_options' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Dict(keys=[],values=[]),node)
        if node.id=='precision' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Constant('none'),node)
        if node.id=='bootstrap_environment' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Constant(False),node)
        if node.id in ('local_io','h3_parallel','shared_exchange','omni_decode') and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Constant(False if node.id in ('local_io','omni_decode') else None),node)
        return node

    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('precision','local_io','h3_parallel','shared_exchange','component_options','omni_decode'):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self,node):
        return None if node.module in ('h3_parallel_config','h3_cuda_options','h3_cuda_precision','framework.vae.omni_decoder') else node

    def visit_BoolOp(self,node):
        self.generic_visit(node)
        is_and=isinstance(node.op,ast.And)
        if any(isinstance(v,ast.Constant) and bool(v.value)!=is_and for v in node.values):
            return ast.copy_location(ast.Constant(not is_and),node)
        values=[v for v in node.values if not isinstance(v,ast.Constant)]
        if not values:return ast.copy_location(ast.Constant(is_and),node)
        if len(values)==1:return values[0]
        node.values=values
        return node

    def visit_UnaryOp(self,node):
        self.generic_visit(node)
        if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant):
            return ast.copy_location(ast.Constant(not node.operand.value),node)
        return node

    def visit_If(self,node):
        # The installer adds this opt-in output hook; an empty enabled set
        # must leave the pre-profile CUDA installer unchanged.
        if (isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Constant)
                and node.test.left.value=='XGEN_H3_FUSED_OUTPUT_ADALN'
                and len(node.test.ops)==1 and isinstance(node.test.ops[0],ast.In)
                and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Name)
                and node.test.comparators[0].id=='enabled'):
            return None
        self.generic_visit(node)
        value=constant(node.test)
        return node if value is None else node.body if value else node.orelse

    def visit_IfExp(self,node):
        self.generic_visit(node)
        value=constant(node.test)
        return node if value is None else node.body if value else node.orelse

    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and (
                isinstance(call.func,ast.Name) and call.func.id in ('add_arguments','add_cuda_options','resolve','commit_environment','validate_options')
                or isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name)
                and call.func.value.id=='H3ParallelConfig' and call.func.attr=='from_args'):
            return None
        if (isinstance(call,ast.Call) and call.args and isinstance(call.args[0],ast.Constant)
                and call.args[0].value in {'--'+name for name in FLAGS}):
            return None
        return self.generic_visit(node)

    def visit_List(self,node):
        self.generic_visit(node)
        node.elts=[item for item in node.elts if not (isinstance(item,ast.Constant) and item.value in FLAGS)]
        return node


@pytest.mark.parametrize('name',('config.py','load_pipe.py','infer_tools.py','h3_cuda.py'))
def test_new_flags_off_preserve_previous_core(name):
    path='x_diffusers/x_diffusers/adaptor/'+name
    result=subprocess.run(['git','show',BASELINE+':'+path],cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    if result.returncode: pytest.skip('Reference Git history is not available in this source-only checkout')
    before=DisabledPrecision().visit(ast.parse(result.stdout))
    after=DisabledPrecision().visit(ast.parse((ROOT/path).read_text(encoding='utf-8')))
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)


def test_tp_cp_off_preserves_public_transformer_core():
    path='x_diffusers/x_diffusers/framework/transformer/minimax_h3.py'
    result=subprocess.run(['git','show','6081426:'+path],cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    if result.returncode:pytest.skip('Reference Git history is not available in this source-only checkout')
    before=DisabledPrecision().visit(ast.parse(result.stdout))
    after=DisabledPrecision().visit(ast.parse((ROOT/path).read_text(encoding='utf-8')))
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
