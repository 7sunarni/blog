# Scale K8s deployment with nginx-ingress-controller metrics

## Step by step with self-checking

### 1. Ingress-NGINX Controller 
Ingress-NGINX Controller have two versions, which maintained by [nginx team](https://github.com/nginxinc/kubernetes-ingress) and [K8s team](https://github.com/kubernetes/ingress-nginx), here we use [K8s team](https://github.com/kubernetes/ingress-nginx)
- Follow the [official document](https://kubernetes.github.io/ingress-nginx/deploy/), install the Ingress-NGINX Controller and make sure the ingress-nginx Pod is running. Notice here you need to enable metrics collection, if you install it with helm, [set this](https://github.com/kubernetes/ingress-nginx/tree/main/charts/ingress-nginx#prometheus-metrics), and after you enable the metrics collection, you can find args in pod spec.
```yaml
# ingress-nginx pod spec
...
metadata:
  annotations:
    prometheus.io/port: "20002"
    prometheus.io/scrape: "true"
...
```
- Follow [official document](https://kubernetes.io/docs/concepts/services-networking/ingress/) deploy a test deployment, configure its service and ingress rule. And make some http requests through ingress
- check the ingress-nginx already collect the request metrics 
```sh
curl http://$ingress_nginx_pod_ip:$metrics_port/metrics | grep "nginx_ingress_controller_requests"

# if you see the counter not 0, it's ok here
# HELP nginx_ingress_controller_requests The total number of client requests.
# 100# TYPE nginx_ingress_controller_requests counter
# 5115 nginx_ingress_controller_requests{canary="",controller_class="k8s.io/ingress-nginx",controller_namespace="kube-system",controller_pod="",ingress="",method="GET",namespace="default",path="",service="",status="200"} 747

```
### 2. Prometheus
- Follow the [official helm document](https://github.com/prometheus-community/helm-charts/tree/main/charts/prometheus#install-chart) install prometheus, make sure prometheus-server pod is running.

### 3. Prometheus-Adapter
- Follow the [official helm document](https://github.com/prometheus-community/helm-charts/blob/main/charts/prometheus-adapter/README.md#install-helm-chart) install prometheus-adapter, make sure prometheus-adapter pod is running. Notice here you need configure prometheus-adapter connect to prometheus-server with assign its args ``.
```yaml
# prometheus-adapter pod spec
containers:
  - args:
    - /adapter
    - --secure-port=6443
    - --cert-dir=/tmp/cert
    - --prometheus-url=http://prometheus-server.kube-system.svc.cluster.local:80 # prometheus-server address
    - --metrics-relist-interval=1m
    - --v=8 # increase log verbosity
    - --config=/etc/adapter/config.yaml
```
- check K8s can correctly get prometheus-adapter metrics
```shell
kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1/namespaces/*/metrics/nginx_ingress_controller_requests'
# if you get some data, its ok
```
- add nginx_ingress_controller_requests_rate metrics to prometheus-adapter configmap, then restart prometheus-adapter pod
```yaml
# prometheus-adapter configmap
    - seriesQuery: '{__name__=~"^nginx_ingress_controller_requests.*",namespace!=""}'
      seriesFilters: []
      resources:
        template: <<.Resource>>
        overrides:
          namespace:
            resource: "namespace"
      name:
        matches: ""
        as: "nginx_ingress_controller_requests_rate"
      metricsQuery: round(sum(rate(<<.Series>>{<<.LabelMatchers>>}[5m])) by (<<.GroupBy>>), 1)
```
- check nginx_ingress_controller_requests metrics can be allocate correct
```shell
kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1/namespaces/*/metrics/nginx_ingress_controller_requests_rate'
# if you get some data, its ok
```
### HPA
- Follow [official document](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/) deploy hpa

### Config ingress-nginx metrics hpa
- create hpa rule
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: # hpa name
  namespace: default
spec:
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 10
        periodSeconds: 30
  maxReplicas: 8
  metrics:
  - type: Object
    object:
      metric:
        name: nginx_ingress_controller_requests_rate
      describedObject:
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        name: # ingress name
      target:
        type: Value
        value: 10
  minReplicas: 3
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: # deploy name
```
