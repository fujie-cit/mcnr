# MCNR

signal processing based multi channel noise reduction

## Installation

from the GitHub repository:

```bash
pip install git+https://github.com/fujie-cit/mcnr.git
```

from the local repository:

```bash
pip install -e .
```

note: -e option is suggested to install the package in the editable mode.

## Usage

The following command will apply the noise reduction to the input.wav and save the result to the output.wav.

The input.wav shoudl be 16kHz, 16bit, multi-channel wav file.

```bash
mcnr -i input.wav output.wav
```

## Reference

The algorithm is based on the following paper:
Aoki M, Aoki S, Okamoto M. Sound source segregation using inter-channel phase and intensity differences. Proc Acoust Soc Japan Vol. 1-7-13, p 489–490, 1996.
