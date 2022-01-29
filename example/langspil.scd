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
~synth.set(\drive, [1,1,4]);
~synth.set(\lr, 3e-3);
~synth.set(\reg, 1e-3);
~synth.set(\pred_gain, [1,1,0]);
~synth.set(\err_gain, [0.5,0.5,1]);
)

~synth.free

s.quit
s.scope

s.scope