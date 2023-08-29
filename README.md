# Dog Bark Detection on Real-time Audio Stream
A containerised transformer-based ML system for detecting "dog barks" in real-time audio stream.

### Build and Run Docker

`docker build . -t dog_bark_detect`

`docker run --device /dev/snd  dog_bark_detect`
