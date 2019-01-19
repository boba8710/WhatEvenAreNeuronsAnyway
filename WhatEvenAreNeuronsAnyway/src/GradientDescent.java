
public class GradientDescent {
	NeuralNetwork internalNetwork;
	double epsilon;
	public GradientDescent(int inputLayerSize, int outputLayerSize, int hiddenLayerCount, int neuronsPerHiddenLayer, double epsilon){
		this.internalNetwork = new NeuralNetwork(inputLayerSize, outputLayerSize);
		for(int i = 0; i < hiddenLayerCount; i++){
			try {
				this.internalNetwork.addHiddenLayer(neuronsPerHiddenLayer);
			} catch (InputWebException e) {
				e.printStackTrace();
			}
		}
		try {
			this.internalNetwork.createInputWeb();
		} catch (InputWebException e) {
			e.printStackTrace();
		}
		this.epsilon = epsilon;
		this.internalNetwork.randomizeAllHiddenLayerBiases();
		this.internalNetwork.randomizeAllHiddenLayerWeights();
		System.out.println("Network instantiated");
	}
	private double computeScore(double[] networkOutput, double[] actualOutput){
		assert networkOutput.length==actualOutput.length;
		double score = 0;
		for(int i = 0; i < networkOutput.length; i++){
			score+=Math.abs(networkOutput[i]-actualOutput[i]);
		}
		score=score/networkOutput.length;
		score=1/score;
		return score;
	}
	private double computeScoreForInputs(double[][] inputs, double[][] outputs){
		System.out.println("Scoring input set");
		double averageScore = 0;
		for(int j = 0; j < inputs.length; j++){
			System.out.println("Input "+j+" scored");
			internalNetwork.initializeInputLayer(inputs[j]);
			internalNetwork.computeOutput();
			double[] networkOutput = new double[internalNetwork.getOutputLayer().size()];
			for(int i = 0; i < internalNetwork.getOutputLayer().size();i++){
				networkOutput[i] = internalNetwork.getOutputLayer().get(i).getOutput();
			}
			
			double score = computeScore(networkOutput, outputs[j]);
			averageScore+=score;
		}
		averageScore=averageScore/inputs.length;
		return averageScore;
	}
	public NeuralNetwork trainNetwork(double[][] inputs, double[][] outputs, int iterations){
		System.out.println("Starting network training");
		assert inputs.length==outputs.length;
		for(int i = 0; i < iterations; i++){
			double baselineScore=computeScoreForInputs(inputs,outputs);
			System.out.println("Baseline score computed for iteration "+i);
			for(int j = 0; j < internalNetwork.getHiddenLayers().size(); j++){
				System.out.println("Training hidden layer "+j);
				for(int k = 0; k < internalNetwork.getHiddenLayers().get(j).size();k++){
					for(int l = 0; l < internalNetwork.getHiddenLayers().get(j).get(k).getWeights().length; l++){
						double startingWeight = internalNetwork.getHiddenLayers().get(j).get(k).getWeights()[l];
						internalNetwork.getHiddenLayers().get(j).get(k).setSingleWeight(l, startingWeight+this.epsilon);
						double newScore = computeScoreForInputs(inputs,outputs);
						if(newScore > baselineScore){
							baselineScore = newScore;
						}else{
							internalNetwork.getHiddenLayers().get(j).get(k).setSingleWeight(l, startingWeight-this.epsilon);
							newScore = computeScoreForInputs(inputs,outputs);
							if((newScore > baselineScore)){
								baselineScore = newScore;
							}else{
								internalNetwork.getHiddenLayers().get(j).get(k).setSingleWeight(l, startingWeight);
							}
						}
						
					}
				}
			}
		}
		
		return internalNetwork;
	}
}
