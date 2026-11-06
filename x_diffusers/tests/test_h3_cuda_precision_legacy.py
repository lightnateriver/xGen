"""Prove the new opt-in branches leave the previous default core AST intact."""
import ast
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='6c00ee73c70be190bb5f3851c03ad232ef0a0032'
FLAGS={'h3_cuda_precision':'none'}


def constant(test):
    if any(isinstance(node,(ast.Name,ast.Call,ast.Attribute,ast.Subscript)) for node in ast.walk(test)):
        return None
    return bool(eval(compile(ast.Expression(test),'<constant>','eval'),{'__builtins__':{}}))


class DisabledPrecision(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)==3
                and isinstance(node.args[1],ast.Constant) and node.args[1].value in FLAGS):
            return ast.copy_location(ast.Constant(FLAGS[node.args[1].value]),node)
        return node

    def visit_Name(self,node):
        if node.id=='precision' and isinstance(node.ctx,ast.Load):
            return ast.copy_location(ast.Constant('none'),node)
        return node

    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='precision':
            return None
        return self.generic_visit(node)

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
