# Scale K8s deployment with nginx-ingress-controller metrics

## Step by step with self-checking

### 1. Ingress-Nginx Controller 
Ingress-Nginx Controller have two version, which matained by [nginx team](https://github.com/nginxinc/kubernetes-ingress) and [K8s team](https://github.com/kubernetes/ingress-nginx), here we use [K8s team](https://github.com/kubernetes/ingress-nginx)
- Follow the [offical document](https://kubernetes.github.io/ingress-nginx/deploy/), install the Ingress-Nginx Controller and make sure ingress-nginx pod running. Notice here you need to enable metrics collect, if you install it with helm, [set this)[https://github.com/kubernetes/ingress-nginx/tree/main/charts/ingress-nginx#prometheus-metrics], and after you enable the metrics collcate, you can find args in pod spec.
```yaml
# ingress-nginx pod spec
...
metadata:
  annotations:
    prometheus.io/port: "20002"
    prometheus.io/scrape: "true"
...
```
- Follow [offical document](https://kubernetes.io/docs/concepts/services-networking/ingress/) deploy a test deployment, configure its service and ingress rule. And make some http requests through ingress
- check the ingress-nginx already collect the request metrics 
```sh
curl http://$ingress_nginx_pod_ip:$metrics-port/metrics | grep "nginx_ingress_controller_requests"

# if you see the counter not 0, it's ok here
# HELP nginx_ingress_controller_requests The total number of client requests.
# 100# TYPE nginx_ingress_controller_requests counter
# 5115 nginx_ingress_controller_requests{canary="",controller_class="k8s.io/ingress-nginx",controller_namespace="kube-system",controller_pod="",ingress="",method="GET",namespace="default",path="",service="",status="200"} 747

```
### 2. Prometheus
- Follow the [offical helm document](https://github.com/prometheus-community/helm-charts/tree/main/charts/prometheus#install-chart) install prometheus, make sure prometheus-server pod is running.

### 3. Prometheus-Adapter
- Follow the [offical helm document](https://github.com/prometheus-community/helm-charts/blob/main/charts/prometheus-adapter/README.md#install-helm-chart) install prometheus-adapter, make sure prometheus-adapter pod is running. Notice here you need configure prometheus-adapter connect to prometheus-server with assign its args ``.
```yaml
# prometheus-adapter pod spec
containers:
  - args:
    - /adapter
    - --secure-port=6443
    - --cert-dir=/tmp/cert
    - --prometheus-url=http://prometheus-server.kube-system.svc.cluster.local:80 # prometheus-server address
    - --metrics-relist-interval=1m
    - --v=4
    - --config=/etc/adapter/config.yaml
```
- check k8s can correct get prometheus-adapter metrics
```shell
kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1/namespaces/*/metrics/nginx_ingress_controller_requests
# if you get some data, its ok
```
- add nginx_ingress_controller_requests_rate metrics to prometheus-adapter, then restart 
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
kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1/namespaces/*/metrics/nginx_ingress_controller_requests_rate
# if you get some data, its ok
```
### HPA
- Follow [offical document](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/) deploy hpa

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
