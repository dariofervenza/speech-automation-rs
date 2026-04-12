# Speech automation Rust

Load audio wav, resample it and extract speech using Nvidia canary 1b v2


## Future goals

What I want to build is an app that:

- Continously listens to my microphone

- Extract many audio temporal windows

- Performs speech recognition

- Searches for specific commands that I will code

- Executes the command when a match is found

- Registers in MongoDb each interaction and the parsed speech

- Containerized application, easy to deply


## Models

- [Canary v2 ONNX](https://huggingface.co/istupakov/canary-1b-v2-onnx)

## Microphone in wsl

- [usbipd-win](https://github.com/dorssel/usbipd-win)
- [usbipd wiki](https://github.com/dorssel/usbipd-win/wiki/WSL-support)
- [configure openocd to allow non root user access to usb device](https://forgge.github.io/theCore/guides/running-openocd-without-sudo.html)
- [alsa](https://oneuptime.com/blog/post/2026-03-02-how-to-install-and-configure-alsa-on-ubuntu/view)

I wasn't sure what I needed so I got alsa-utils, alsa-firmware, alsa-lib and alsa-oss

lspci | grep -i audio

https://medium.com/@searcheulong/fix-audio-not-working-on-ubuntu-24-04-3-at-least-in-my-case-fc359073f433
https://research.wmz.ninja/articles/2017/11/setting-up-wsl-with-graphics-and-audio.html
sudo apt install meson ninja-build build-essential libcap-dev

- pulse audio install (17.0)

meson setup build \
    -Dbluez5=disabled \
    -Dudev=disabled \
    -Dsystemd=disabled \
    -Dasyncns=disabled \
    -Ddbus=disabled \
    -Ddaemon=true
https://askubuntu.com/questions/1347090/how-to-install-newest-pulseaudio-on-ubutu-20-04
 sudo apt-get install check
 sudo apt-get install doxygen

 ninja -C build
 sudo ninja -C build install

 stop-process -name "pulseaudio" -Force -ErrorAction SilentlyContinue
 .\pulseaudio.exe -n -F ..\etc\pulse\default.pa --exit-idle-time=-1
 https://github.com/microsoft/WSL/issues/2658
 export PULSE_SERVER=$(ip route show | grep default | awk '{print $3}')
 pactl info


- TEST PULSE AUDIO MIC: parecord --channels=1 --format=s16le --rate=44100 test.wav

## TODOS:

- Do not hardcode --> Move everything to config/specs files

- Add tests

- Expand with more models?


# Why I am creating this

I wanted to practice Rust but my profile is focused on Python.

I could start a pytorch project or anything python-based.

However, using rust I'm going to achieve many goals:

- Handle problems in the hard way + learn more Rust

- Learn about model preprocessing tehcniques + SOTA models (eg. audio preprocessing with fbanks)

- Maintain contact with python projects due to the rust complexity and the necessity to reverse-engineer model architectures

- Practice MongoDB, as it will my main database
