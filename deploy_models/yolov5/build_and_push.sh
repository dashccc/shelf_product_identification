#!/bin/bash
image=$1
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
# fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [[ $region =~ ^cn.* ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:latest"
    registry_id="727897471807"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com.cn"
elif [[ $region = "ap-east-1" ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
    registry_id="871362719292"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
    registry_id="763104351884"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com"
fi

# if [ $? -ne 0 ]
# then
#     aws ecr create-repository --repository-name "${image}" > /dev/null
# fi
aws ecr create-repository --repository-name "${image}" > /dev/null

$(aws ecr get-login --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids ${registry_id} --region ${region} --no-include-email)

docker build -t ${image} -f Dockerfile .
docker tag ${image} ${fullname}

docker push ${fullname}
