


run:
    echo "hello"


create-env:
    conda create --name mlp-cw4 python=3.7
    conda install pytorch==1.9.0 \
                torchvision==0.10.0 \
                cudatoolkit=11.3 \
                -c pytorch -c conda-forge