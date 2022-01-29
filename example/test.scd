(
s.options.sampleRate = 48000;
s.options.blockSize = 64;
s.boot
)

~synth.set(\freq, 450)
~synth.set(\lr, 3e-3)
~synth.set(\reg, 1e-3)


(
~synth = {
    var freq = \freq.kr(100, 0.1);
    var lr = \lr.kr(1e-3);
    var reg = \reg.kr(1e-3);
    var src = (Mix.ar(SinOsc.ar([freq, freq+10, freq*2+1, freq*5-3]))
        + (WhiteNoise.ar*1e-2)
        + SoundIn.ar(0)
        )/10;
    // var src = SoundIn.ar(0);
    var pred = Predictor.ar(src, K2A.ar(lr), K2A.ar(reg));
    var err = (pred-src);

    // Limiter.ar([Delay1.ar(src), pred])
    Limiter.ar([Delay1.ar(src)-src, err])
    // Pan2.ar(err)
    // Pan2.ar(pred)

}.play
)


~synth.set(\drive, 32);
~synth.set(\lr, 1e-2)
~synth.set(\reg, 1e-2)


~synth.free

s.quit
s.scope

s.scope