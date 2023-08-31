# Dog Bark Detection on Real-time Audio Stream
A containerised transformer-based ML system for detecting "dog barks" in real-time audio stream.

Model Weights : [Google Drive](https://drive.google.com/file/d/1plEa_9DmXzh0Q_giTHI9wl1CYbXbIWn5/view?usp=sharing)

### Build and Run Docker

`docker build . -t dog_bark_detect`

`docker run --device /dev/snd  dog_bark_detect`
