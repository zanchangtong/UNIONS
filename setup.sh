
# python=3.8
# fairseq commit id: d3890e593398c485f6593ab8512ac51d37dedc9c

git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout d3890e593398c485f6593ab8512ac51d37dedc9c
pip install -e .

pushd ../exps
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
popd
