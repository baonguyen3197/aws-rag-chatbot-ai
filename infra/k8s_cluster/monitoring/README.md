helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm search repo loki

helm install loki grafana/loki -n loki --create-namespace -f loki1.yaml
helm install grafana-alloy grafana/alloy -n loki --create-namespace -f alloy1.yaml 
helm install grafana grafana/grafana -n grafana --create-namespace -f grafana.yaml
helm install prometheus prometheus-community/prometheus --namespace prometheus --create-namespace

kubectl patch svc grafana -n grafana -p '{\"spec\":{\"type\": \"LoadBalancer\"}}'

http://prometheus-server.monitoring.svc.cluster.local
http://loki-gateway.loki.svc.cluster.local

1860