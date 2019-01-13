import java.util.ArrayList;
import java.util.Random;


public class GeneticTraining {
	private int populationSize, preserveTopNIndividuals;
	private double mutationChance, populationDieOffPercent;
	private int inputCount; 
	private int outputCount;
	private int hiddenLayerCount; 
	private int neuronsPerHiddenLayer;
	private NeuralNetwork[] population;
	private static void createInitialPopulation(NeuralNetwork[] population, int populationSize, int inputCount, int outputCount ,int hiddenLayerCount, int neuronsPerHiddenLayer){
		population = new NeuralNetwork[populationSize];
		for(NeuralNetwork nN : population){
			nN = new NeuralNetwork(inputCount, outputCount);
			for(int i=0; i < hiddenLayerCount;i++){
				try {
					nN.addHiddenLayer(neuronsPerHiddenLayer);
				} catch (InputWebException e) {
					e.printStackTrace();
				}
			}
			try {
				nN.createInputWeb();
			} catch (InputWebException e) {
				e.printStackTrace();
			}
			nN.randomizeAllHiddenLayerBiases();
			nN.randomizeAllHiddenLayerWeights();
		}
	}
	public GeneticTraining(int populationSize, double mutationChance, double populationDieOffPercent, int preserveTopNIndividuals, int inputCount, int outputCount ,int hiddenLayerCount, int neuronsPerHiddenLayer){
		this.mutationChance = mutationChance;
		this.populationDieOffPercent=populationDieOffPercent;
		this.populationSize=populationSize;
		assert (1/populationDieOffPercent)%populationSize==0;
		this.preserveTopNIndividuals=preserveTopNIndividuals;
		this.inputCount = inputCount;
		this.outputCount=outputCount;
		this.hiddenLayerCount = hiddenLayerCount;
		this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
		createInitialPopulation(this.population, populationSize, inputCount, outputCount, hiddenLayerCount,neuronsPerHiddenLayer);
	}
	/**
	 * Scores the population. The result will be that all networks in the population
	 * will have their scores updated.
	 * @assert the length of each input array must equal the number of neurons in the input layer
	 * @assert the length of each output array must equal the number of neurons in the output layer
	 * @param inputs
	 * @param outputs
	 */
	private static void scorePopulation(double[][] inputs, double[][] outputs, int inputCount, int outputCount, NeuralNetwork[] population){
		assert inputs[0].length==inputCount;
		assert outputs[0].length==outputCount;
		for(NeuralNetwork network : population){
			double networkScore = 0;
			for(int i = 0; i < inputs.length; i++){
				network.initializeInputLayer(inputs[i]);
				network.computeOutput();
				ArrayList<Neuron> output = network.getOutputLayer();
				double[] networkActualOutput = new double[output.size()];
				for(int j = 0; j < output.size(); j++){
					networkActualOutput[j]=output.get(j).getOutput();
				}
				double caseScore = 0;
				for(int j = 0; j < outputs[i].length;j++){
					caseScore+=outputs[i][j]-networkActualOutput[j];
				}
				caseScore = 1/Math.pow(caseScore, 2);
				networkScore+=caseScore;
			}
			networkScore=networkScore/inputs.length;
			network.setNetworkScore(networkScore);
		}
	}
	/**
	 * Sorts the population in place
	 * @param population
	 */
	private static void sortPopulation(NeuralNetwork[] population){
		Quicksort.quickSort(population, 0, population.length-1);
	}
	/**
	 * performs one sweep of mutation over the population. Each weight of each neuron in each hiddenLayer
	 * of each individual has a mutationChance chance of being randomized. Setting a higher mutationChance
	 * results in more exploration done by the network.
	 * @param population
	 * @param neuronsPerHiddenLayer
	 * @param mutationChance
	 * @param preserveTopNIndividuals
	 */
	private static void mutatePopulation(NeuralNetwork[] population,int neuronsPerHiddenLayer, double mutationChance, int preserveTopNIndividuals){
		Random r = new Random();
		int i = 1;
		for(NeuralNetwork network : population){
			if(i>preserveTopNIndividuals){
				
				ArrayList<ArrayList<Neuron>> hiddenLayers = network.getHiddenLayers();
				for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
					int totalWeights = hiddenLayer.get(0).getInputCount();
					for(Neuron n : hiddenLayer){
						for(int j = 0; j<totalWeights;j++){
							double randDouble = r.nextDouble();
							if(randDouble<mutationChance){
								n.randomizeSingleWeight(i);
							}
						}
						if(r.nextDouble()<mutationChance){
							n.setRandomBias();
						}
					}
				}
				
			}
			i++;
		}
	}
	private static NeuralNetwork crossbreedIndividuals(NeuralNetwork individualA, NeuralNetwork individualB){
		ArrayList<ArrayList<Neuron>> hiddenLayersA = individualA.getHiddenLayers();
		ArrayList<ArrayList<Neuron>> hiddenLayersB = individualB.getHiddenLayers();
		NeuralNetwork childNetwork = individualA;
		
		Random r = new Random();
		if(Globals.CROSSBREEDING_FINE_GRANULARITY){//Fine Granularity, crossbreeds weights
			for(int i = 0; i< hiddenLayersA.size(); i++){//layers
				for(int j = 0; j<hiddenLayersA.get(0).size();j++){//neurons
					Neuron childNeuron = hiddenLayersA.get(i).get(j);
					childNeuron.setInputCount(hiddenLayersA.get(i).get(j).getInputCount());
					for(int k = 0; k<hiddenLayersA.get(0).get(0).getInputCount(); k++){//weights
						if(r.nextBoolean()){
							childNeuron.setSingleWeight(k, hiddenLayersA.get(i).get(j).getSingleWeight(k));
						}else{
							childNeuron.setSingleWeight(k, hiddenLayersB.get(i).get(j).getSingleWeight(k));
						}
					}
					childNetwork.setNeuron(i, j, childNeuron);
				}
			}
		}else{//Coarse granularity, crossbreeds entire neurons
			for(int i = 0; i< hiddenLayersA.size(); i++){//layers
				for(int j = 0; j<hiddenLayersA.get(0).size();j++){//neurons
					Neuron childNeuron = hiddenLayersA.get(i).get(j);
					childNeuron.setInputCount(hiddenLayersA.get(i).get(j).getInputCount());
					if(r.nextBoolean()){
						childNetwork.setNeuron(i, j, hiddenLayersA.get(i).get(j));
					}else{
						childNetwork.setNeuron(i, j, hiddenLayersB.get(i).get(j));
					}
					
				}
			}
		}
		return childNetwork;
	}
	public void runGeneration(double[][] testingInputArray, double[][] testingOutputArray){
		scorePopulation(testingInputArray, testingOutputArray, inputCount, outputCount, population);
		sortPopulation(population);
		mutatePopulation(population, neuronsPerHiddenLayer, mutationChance, preserveTopNIndividuals);
		
		
	}
}
