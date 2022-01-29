ServerOptions.devices;

(
Server.default.options.inDevice_("K-Mix");
Server.default.options.outDevice_("K-Mix");
s.options.numInputBusChannels = 8;
s.options.numOutputBusChannels = 8;
s.options.sampleRate = 48000;
s.options.blockSize = 16;
s.boot
)

~pup_buses = [5,6,7];


// 1 predictor for mix of 3 pickups
(
~synth = {
    var lr = \lr.kr(1e-3);
    var reg = \reg.kr(1e-3);
    var src = Mix.ar(SoundIn.ar(~pup_buses)) * \drive.kr(1, 0.1);

    var pred = Predictor.ar(src, K2A.ar(lr), K2A.ar(reg));
    var err = (pred-src);

    Limiter.ar(Mix.ar([
        pred * \pred_gain.kr(0, 0.1),
        err * \err_gain.kr(0, 0.1)
    ]), \limit.kr(0.9, 0.1), 1e-3);

}.play
)

//feedback suppression

(
~synth.set(\drive, 4);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, 0);
~synth.set(\err_gain, 1);
)

//incomplete fb suppression with high gain and learning rate
(
~synth.set(\drive, 32);
~synth.set(\lr, 1e-2);
~synth.set(\reg, 1e-2);
~synth.set(\pred_gain, 0);
~synth.set(\err_gain, 1);
)

//incomplete fb suppression with low learning rate
(
~synth.set(\drive, 5);
~synth.set(\lr, 3e-6);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, 0);
~synth.set(\err_gain, 1);
)

//smoothed feedback with predictable signal only
(
~synth.set(\drive, 1);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, 1);
~synth.set(\err_gain, 0);
)


// predictor per-pickup
(
~synth = {
    var lr = \lr.kr(1e-3);
    var reg = \reg.kr(1e-3);
    var src = SoundIn.ar(~pup_buses) * \drive.kr([1,1,1], 0.1);

    var pred = Predictor.ar(src, K2A.ar(lr), K2A.ar(reg));
    var err = (pred-src);

    Limiter.ar(Mix.ar([
        pred * \pred_gain.kr([0,0,0], 0.1),
        err * \err_gain.kr([0,0,0], 0.1)
    ].flatten), \limit.kr(0.9, 0.1), 1e-3);

}.play
)

// fb suppression for melody string only
(
~synth.set(\drive, [1,1,12]);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, [1,1,0]);
~synth.set(\err_gain, [0.5,0.5,1]);
)


// incorporated with other effects
(
~synth = {
    var lr = \lr.kr(1e-3);
    var reg = \reg.kr(1e-3);
    var src = SoundIn.ar(~pup_buses) * \drive.kr([1,1,1], 0.1);

    var pred = Predictor.ar(src, K2A.ar(lr), K2A.ar(reg));
    var err = (pred-src);

    Limiter.ar(Mix.ar([
        LPF.ar(
            FreqShift.ar(pred, \shift.kr([0,0,0], 0.1)),
            \lpf.kr([10000,10000,10000], 0.1))
        * \pred_gain.kr([0,0,0], 0.1),
        err+CombC.ar(err, 10,
            SampleDur.ir * 4 + \delay.kr([0,0,0], 1),
            \decay.kr([0,0,0], 1))
        * \err_gain.kr([0,0,0], 0.1)
    ].flatten), \limit.kr(0.9, 0.1), 1e-3);

}.play
)

// shifters on the drone strings
(
~synth.set(\drive, 2*[1,1,8]);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, [1,1,0]);
~synth.set(\err_gain, [4,4,1]);
~synth.set(\shift, [4,-5,0]);
~synth.set(\lpf, [1000, 1500, 10000]);
~synth.set(\delay, [0,0,0]);
~synth.set(\decay, [0,0,0]);
)

// delays
(
~synth.set(\drive, [1,1,16]);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, [1,1,0]);
~synth.set(\err_gain, [4,4,1]);
~synth.set(\shift, [0,0,0]);
~synth.set(\lpf, [1000, 1500, 10000]);
~synth.set(\delay, [0.59, 0.46, 0.3]);
~synth.set(\decay, [6, 8, 4]);
)

~synth.free

s.quit
s.scope

s.scope