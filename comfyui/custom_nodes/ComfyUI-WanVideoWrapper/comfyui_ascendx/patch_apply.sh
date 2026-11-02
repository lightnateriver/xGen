\cp patches/model.patch ../
\cp patches/nodes_model_loading.patch ../
\cp patches/vae_decode_multi.patch ../
cd ../
git apply model.patch
git apply nodes_model_loading.patch
git apply vae_decode_multi.patch
