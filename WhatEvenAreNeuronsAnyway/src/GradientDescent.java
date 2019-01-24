import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class GradientDescent {
	NeuralNetwork internalNetwork;
	NeuralNetwork newNetwork;
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
	private int totalScored = 0;
	private double computeScoreForInputs(double[][] inputs, double[][] outputs){
		System.out.println("Scoring input set");
		double averageScore = 0;
		for(int j = 0; j < inputs.length; j++){
			internalNetwork.initializeInputLayer(inputs[j]);
			internalNetwork.computeOutput();
			double[] networkOutput = new double[internalNetwork.getOutputLayer().size()];
			for(int i = 0; i < internalNetwork.getOutputLayer().size();i++){
				networkOutput[i] = internalNetwork.getOutputLayer().get(i).getOutput();
			}
			
			double score = computeScore(networkOutput, outputs[j]);
			averageScore+=score;
			totalScored+=1;
		}
		System.out.println("Scored inputs: "+totalScored);
		averageScore=averageScore/inputs.length;
		return averageScore;
	}
	
	
	private void tweakWeights(double[][] inputs, double[][] outputs, Neuron neuron, double baselineScore){
		for(int i = 0; i < neuron.getWeightCount(); i++){
			double startingWeight = neuron.getSingleWeight(i);
			neuron.setSingleWeight(i, startingWeight+this.epsilon);
			double newScore = computeScoreForInputs(inputs,outputs);
			if(newScore > baselineScore){
				baselineScore = newScore;
			}else{
				neuron.setSingleWeight(i, startingWeight-this.epsilon);
				newScore = computeScoreForInputs(inputs,outputs);
				if((newScore > baselineScore)){
					baselineScore = newScore;
				}else{
					neuron.setSingleWeight(i, startingWeight);
				}
			}
		}
	}
	//Next on the paralellization chopping block
	public NeuralNetwork trainNetwork(double[][] inputs, double[][] outputs, int iterations){
		System.out.println("Starting network training");
		assert inputs.length==outputs.length;
		for(int i = 0; i < iterations; i++){//iterations cannot be multithreaded
			NeuralNetwork networkCopy = null;
			try {
				networkCopy = internalNetwork.deepCopy();
			} catch (InputWebException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			double baselineScore=computeScoreForInputs(inputs,outputs);//baseline score must be computed once per iteration
			for(ArrayList<Neuron> hiddenLayer : networkCopy.getHiddenLayers()){ //Layers should not be multithreaded due to relying on prior layers for output
				final ExecutorService executor = Executors.newCachedThreadPool();
				final List<Future<?>> futures = new ArrayList<>();
				for(Neuron neuron:hiddenLayer) {
					Future<?> future = executor.submit(() -> {
						tweakWeights(inputs, outputs, neuron, baselineScore);
					});
					futures.add(future);
				}
				try {
					for (Future<?> future : futures) {
						future.get();
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			this.internalNetwork.copyFromNetwork(networkCopy);
		}
		return internalNetwork;
	}
}
