Sources: NVlabs/Sana, commit 781e8341fb3b29d3aea5d50265772e9619340b66,
models/Open-H3-Regenerate-2K/open_h3/{adapters,geometry,sigmas,upscaler,upscaler_node,vae_fast}.py.
Retrieved 2026-10-31. Source snapshot retained with local changes: vae_fast.py handles ranks with no spatial tiles.
upscaler.py also scopes import shims to module loading and restores prior sys.modules entries.
The NPU adapter and resource policy live outside vendor/.
upscaler_node.py originates from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT);
the code license is preserved in LICENSE-upscaler (retrieved from its upstream
repository on 2026-10-31). This is distinct from the upscaler weight license.
Model weights have their own licenses; see the source model README.
