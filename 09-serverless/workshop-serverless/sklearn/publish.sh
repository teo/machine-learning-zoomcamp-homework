#!/usr/bin/env bash

ECR_URL=963324162193.dkr.ecr.eu-north-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda

REMOTE_IMAGE_TAG="${REPO_URL}:v1"
LOCAL_IMAGE=churn-prediction-lambda

docker build -t ${LOCAL_IMAGE} .

aws ecr get-login-password --region eu-north-1 \
| docker login --username AWS --password-stdin ${ECR_URL}


docker tag $LOCAL_IMAGE ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}
echo "Image pushed to ${REMOTE_IMAGE_TAG}"