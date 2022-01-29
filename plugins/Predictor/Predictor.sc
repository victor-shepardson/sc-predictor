Predictor : UGen {
	*ar { |input, lr=1e-3, reg=0|
		^this.multiNew('audio', input, lr, reg);
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}
