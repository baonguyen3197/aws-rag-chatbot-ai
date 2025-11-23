```Aws ArgoCD Repository ```
============================
Get AWS K8s config
```bash
aws eks --region <region> update-kubeconfig --name <cluster_name>
```

============================
```bash
kubectl create namespace argocd
```

============================
```powershell
kubectl create secret docker-registry regcred `
  --docker-server=https://index.docker.io/v1/ `
  --docker-username=<DOCKERHUB_USER> `
  --docker-password=<DOCKERHUB_PASS> `
  --docker-email=<you@example.com> `
  -n argocd
```

```bash
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<DOCKERHUB_USER> \
  --docker-password=<DOCKERHUB_PASS> \
  --docker-email=<you@example.com> \
  -n argocd
```

============================
```bash
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

kubectl apply -n argo-rollouts -f https://raw.githubusercontent.com/argoproj/argo-rollouts/stable/manifests/install.yaml
```

============================
```powershell
kubectl patch svc argocd-server -n argocd -p '{\"spec\": {\"type\": \"LoadBalancer\"}}'
kubectl patch svc argocd-server -n argocd -p '{\"metadata\": {\"annotations\": {\"service.beta.kubernetes.io/aws-load-balancer-type\": \"nlb\", \"service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled\": \"true\"}}}'
kubectl get svc -n argocd
```

```bash
kubectl patch svc argocd-server -n argocd -p '{"spec":{"type":"LoadBalancer"}}'
kubectl patch svc argocd-server -n argocd -p '{"metadata":{"annotations":{"service.beta.kubernetes.io/aws-load-balancer-type":"nlb","service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled":"true"}}}'
kubectl get svc -n argocd
```

============================
```powershell
$encodedPassword = kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath="{.data.password}"

[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($encodedPassword))
```

```bash
kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' | base64 -d; echo
```

============================
```bash
kubectl patch cm argocd-cm -n argocd --type merge \
  -p '{"data":{"accounts.nhqb":"apiKey,login","accounts.nhqb.enabled":"true"}}'
kubectl patch cm argocd-rbac-cm -n argocd --type merge \
  -p '{"data":{"policy.csv":"g, nhqb, role:admin\n","policy.default":"role:readonly"}}'
```

```powershell
'{"data":{"accounts.nhqb":"apiKey,login","accounts.nhqb.enabled":"true"}}' | Out-File -FilePath .\argocd-accounts.json -NoNewline -Encoding ascii
'{"data":{"policy.csv":"g, nhqb, role:admin\n","policy.default":"role:readonly"}}' | Out-File -FilePath .\argocd-rbac.json -NoNewline -Encoding ascii

kubectl patch cm argocd-cm -n argocd --type=merge --patch-file .\argocd-accounts.json
kubectl patch cm argocd-rbac-cm -n argocd --type=merge --patch-file .\argocd-rbac.json
```

============================
# Argo Rollouts
============================
```bash
# Install Argo Rollouts CRDs and Controller
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

kubectl apply -f nginx-rollouts.yaml
kubectl argo rollouts get rollout nginx-rollout --watch
kubectl argo rollouts promote nginx-rollout
kubectl argo rollouts history nginx-rollout

kubectl argo rollouts abort nginx-rollout

kubectl argo rollouts dashboard
```
