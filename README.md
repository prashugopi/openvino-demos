# openvino-demos
# Mike Sedillo & Prashanth Gopinatth
## Facial Recognition demo

### To build for CPU
```
docker-compose -f docker-compose.yml -f docker-compose.face-cpu.yml build
docker-compose -f docker-compose.yml -f docker-compose.face-cpu.yml up
```
### To build for GPU

```
docker-compose -f docker-compose.yml -f docker-compose.face-gpu.yml build
docker-compose -f docker-compose.yml -f docker-compose.face-gpu.yml up
```
