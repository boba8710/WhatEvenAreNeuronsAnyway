import java.util.Random;

public class MainClass {
	public static void main(String[] args){
		NeuralNetwork neuralNetwork = new NeuralNetwork(10, 2);
		try {
			neuralNetwork.addHiddenLayer(1000);
			neuralNetwork.addHiddenLayer(1000);
			neuralNetwork.createInputWeb();
		} catch (InputWebException e) {
			e.printStackTrace();
		}
		neuralNetwork.randomizeAllHiddenLayerBiases();
		neuralNetwork.randomizeAllHiddenLayerWeights();
		
		Random r = new Random();
		double[] startingValues = new double[10];
		for(int i = 0; i < startingValues.length; i++){
			startingValues[i]=(r.nextDouble()>0.5 ? 1:-1)*r.nextDouble();
		}
		neuralNetwork.initializeInputLayer(startingValues);
		neuralNetwork.computeOutput();
		Object[] output = neuralNetwork.getOutputLayer().toArray();
		for(Object o : output){
			Neuron n = (Neuron)o;
			System.out.println(n.getOutput());

		}
					
											
	}
}