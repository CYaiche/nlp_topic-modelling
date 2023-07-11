# Test locally 
## Build docker image
docker build -t <\image name> .


##  lists docker images 
docker images
docker ps 
docker ps -a

## start 

docker run -d -p 80:80 <\image-id>
docker run -p 8501:8501 <\image name> 

- p for --publish 

# exec
docker exec -it  <\image-id> /bin/sh

## stop 
docker stop <\image-name>  : will not remove the image only stop 

## delete images 

docker rmi <\image-id>

# view in html 
http://localhost:8501