
export CLOUDSDK_CORE_PROJECT=s1860947-mlpractical
export CLOUDSDK_COMPUTE_REGION=us-west1
export CLOUDSDK_COMPUTE_ZONE=us-west1-b

EXPERIMENT_NAME=$(shell grep -A3 'experiment_name:' ${CONFIG} | head -n1 | cut -d'"' -f 2 |  tr -d '_' | tr '[:upper:]' '[:lower:]')

run:
	echo "You fucking twat"


installEnv:
	conda install pytorch==1.10.2 \
	torchvision  \
	cudatoolkit=10.2 \
	pyyaml \
	numpy \
	matplotlib \
	seaborn \
	pandas \
	-c pytorch 

createVm:
	test -n "$(CONFIG)" # $$CONFIG;
	@echo "creating ${CONFIG} instance on gcloud";
	@echo "experiment_name: ${EXPERIMENT_NAME}";
	gcloud compute instances create $(EXPERIMENT_NAME) \
		--image-family=ubuntu-2004-lts \
		--image-project ubuntu-os-cloud \
		--machine-type n1-standard-4 \
		--create-disk size=20 \
		--accelerator type=nvidia-tesla-k80,count=2 \
		--maintenance-policy TERMINATE --restart-on-failure \
		--preemptible \


sshVm:
	test -n "$(CONFIG)" # $$CONFIG;
	@echo "experiment_name: ${EXPERIMENT_NAME}";
	gcloud compute ssh ${EXPERIMENT_NAME}


removeVm:
	test -n "$(CONFIG)" # $$CONFIG;
	@echo "experiment_name: ${EXPERIMENT_NAME}";
	gcloud compute instances delete ${EXPERIMENT_NAME}

toVm:
	test -n "$(CONFIG)" # $$CONFIG;
	test -n "$(FROM)" # $$CONFIG;
	test -n "$(TO)" # $$CONFIG;

	@echo "experiment_name: ${EXPERIMENT_NAME}";
	gcloud compute scp --recurse ${FROM} ${EXPERIMENT_NAME}:${TO}

fromVm:
	test -n "$(CONFIG)" # $$CONFIG;
	test -n "$(FROM)" # $$CONFIG;
	test -n "$(TO)" # $$CONFIG;

	@echo "experiment_name: ${EXPERIMENT_NAME}";
	gcloud compute scp --recurse ${EXPERIMENT_NAME}:${FROM} ${TO} 

listDisksVm:
	gcloud compute disks list


listVms:
	gcloud compute instances list