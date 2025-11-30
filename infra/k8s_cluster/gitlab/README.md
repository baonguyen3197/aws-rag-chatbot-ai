# aws-argocd-gitlab
============================
## Get AWS K8s config
```bash
aws eks --region <region> update-kubeconfig --name <cluster_name>
aws eks --region ap-northeast-1 update-kubeconfig --name ttb-terraform-k8s-cluster

# Check status:
aws eks --region ap-northeast-1 describe-addon --cluster-name nhqb-terraform-k8s-cluster --addon-name aws-ebs-csi-driver
```
============================
## Install GitLab with Helm
```bash
# bash
helm repo add gitlab https://charts.gitlab.io/
helm repo update
helm upgrade --install gitlab gitlab/gitlab `
  -n gitlab `
  --create-namespace `
  --set global.hosts.domain=nhqb-gitlab.duckdns.org `
  --set global.hosts.externalIP=18.183.105.100 `
  -f ./infra/k8s_cluster/gitlab/gitlab-values.yaml `
  --set certmanager-issuer.email=baonguyen3197@gmail.com `
  --namespace gitlab

helm upgrade --install gitlab gitlab/gitlab \
  -n gitlab \
  --create-namespace \
  -f ./infra/k8s_cluster/gitlab/gitlab-values.yaml \
  --set certmanager-issuer.email=baonguyen3197@gmail.com \
  --set nginx-ingress.controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-scheme"=internet-facing \
  --set global.hosts.domain=nhqb-gitlab.duckdns.org \
  --namespace gitlab
```

## Update StorageClass to gp2
```bash
kubectl apply -f ./infra/k8s_cluster/gitlab/gp2-csi.yaml
kubectl patch storageclass gp2-csi -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

helm upgrade --install gitlab gitlab/gitlab -n gitlab -f ./infra/k8s_cluster/gitlab/gitlab-values.yaml
```

## Apply Ingress
```bash
kubectl apply -f ./infra/k8s_cluster/gitlab/gitlab-ingress.yaml

# kubectl create namespace ingress-nginx
# kubectl apply -f ./infra/k8s_cluster/ingress/nginx-internal-svc.yaml
```

```bash
helm upgrade --install gitlab gitlab/gitlab \
    -n gitlab \
  -f ./gitlab/gitlab-values.yaml
```

## Check status
```bash
helm status gitlab -n gitlab
```

## Get Secret
```bash
kubectl get secret gitlab-gitlab-initial-root-password -n gitlab -ojsonpath='{.data.password}' | base64 --decode ; echo
```

## Uninstall GitLab
```bash
helm uninstall gitlab -n gitlab
kubectl delete namespace gitlab
```

============================
## Deploy NPM Registry
```bash
kubectl apply -f nginx-proxy-manager/npm-values.yaml
kubectl -n npm get svc npm -w
```

============================
## Add Docker Registry Credentials to Gitlab
```powershell
kubectl create secret docker-registry regcred `
  --docker-server=https://index.docker.io/v1/ `
  --docker-username=<DOCKERHUB_USER> `
  --docker-password=<DOCKERHUB_PASS> `
  --docker-email=<you@example.com> `
  -n gitlab
```


ERROR: Job failed (system failure): pod "gitlab/runner-z2s1d3cda-project-2-concurrent-0-555yux86" is disrupted: reason "TerminationByKubelet", message "The node was low on resource: ephemeral-storage. Threshold quantity: 2139512454, available: 979984Ki. Container build was using 6624724Ki, request is 0, has larger consumption of ephemeral-storage. Container helper was using 8Ki, request is 0, has larger consumption of ephemeral-storage. "