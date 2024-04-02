# iptables udp usage

in tcp connection, because tcp has state, so iptables know the packet belonged the connection, and should send it to where.
but in udp, there is no connection. so how iptables deal it.

## use k8s service as example
1. prepare deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: udp-test
  namespace: default
spec:
  progressDeadlineSeconds: 600
  replicas: 2
  selector:
    matchLabels:
      app: udp-test
  revisionHistoryLimit: 10
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: udp-test
    spec:
      containers:
      - command:
        - nc
        - -lku
        - -p
        - 12345
        image: $IMAGE
        imagePullPolicy: IfNotPresent
        name: debug
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      terminationGracePeriodSeconds: 30
```
2. prepare svc
```yaml
apiVersion: v1
kind: Service
metadata:
  name: udp-test
  namespace: default
spec:
  internalTrafficPolicy: Cluster
  ports:
  - name: metrics
    port: 12345
    protocol: UDP
    targetPort: 12345
  selector:
    app: udp-test
  type: ClusterIP
```
3. nc -u 56789 $SVC_IP 12345 # with multiple try, different pod will receivce request.
4. nc -u --source 127.0.0.1 --source-port 56789 172.20.148.205 12345 # with multiple try, only one pod will receivce request.

and set protocol to tcp, it will still random with same source-port .
