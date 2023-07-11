# Test locally 
## Build docker image
docker build .


##  lists docker images 
docker images
docker ps 
docker ps -a

## start 

docker run -d -p 80:80 <\image-id>
- p for --publish 

# exec
docker exec -it  <\image-id> /bin/sh

## stop 
docker stop <\image-name>  : will not remove the image only stop 

## delete images 

docker rmi <\image-id>