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
		this.population = new NeuralNetwork[populationSize];
		for(int i = 0; i < populationSize; i++){
			population[i] = new NeuralNetwork(inputCount, outputCount);
			for(int j=0; j < hiddenLayerCount;j++){
				try {
					population[i].addHiddenLayer(neuronsPerHiddenLayer);
				} catch (InputWebException e) {
					e.printStackTrace();
				}
			}
			try {
				population[i].createInputWeb();
			} catch (InputWebException e) {
				e.printStackTrace();
			}
			population[i].randomizeAllHiddenLayerBiases();
			population[i].randomizeAllHiddenLayerWeights();
		}
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
		int i = 0;
		for(NeuralNetwork network : population){
			if(i>preserveTopNIndividuals){
				ArrayList<ArrayList<Neuron>> hiddenLayers = network.getHiddenLayers();
				for(ArrayList<Neuron> hiddenLayer : hiddenLayers){
					int totalWeights = hiddenLayer.get(0).getWeightCount();
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
	/**
	 * Generates the 'child' of two neural networks. In the case of fine crossbreeding granularity, each weight of each neuron of each parent has a 50% chance of 
	 * being expressed in the child network. In coarse crossbreeding granularity, each neuron in each parent network has a 50% chance of being expressed in the child.
	 * @param individualA
	 * @param individualB
	 * @return the child NeuralNetwork of individualA and individualB
	 */
	private static NeuralNetwork crossbreedIndividuals(NeuralNetwork individualA, NeuralNetwork individualB){
		ArrayList<ArrayList<Neuron>> hiddenLayersA = individualA.getHiddenLayers();
		ArrayList<ArrayList<Neuron>> hiddenLayersB = individualB.getHiddenLayers();
		NeuralNetwork childNetwork = individualA;
		
		Random r = new Random();
		if(Globals.CROSSBREEDING_FINE_GRANULARITY){//Fine Granularity, crossbreeds weights
			for(int i = 0; i< hiddenLayersA.size(); i++){//layers
				for(int j = 0; j<hiddenLayersA.get(0).size();j++){//neurons
					Neuron childNeuron = hiddenLayersA.get(i).get(j);
					childNeuron.setInputCount(hiddenLayersA.get(i).get(j).getWeightCount());
					for(int k = 0; k<hiddenLayersA.get(0).get(0).getWeightCount(); k++){//weights
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
					childNeuron.setInputCount(hiddenLayersA.get(i).get(j).getWeightCount());
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
	//TODO: Why are these static? Can I just move there code into this method (at the cost of readability)?
	//		How about I change them to private void methods? Why not?
	public void runGeneration(double[][] testingInputArray, double[][] testingOutputArray){
		scorePopulation(testingInputArray, testingOutputArray, inputCount, outputCount, population);
		sortPopulation(population);
		mutatePopulation(population, neuronsPerHiddenLayer, mutationChance, preserveTopNIndividuals);
		NeuralNetwork[] newPopulation = new NeuralNetwork[population.length];
		
		for(int i = 0; i < population.length*populationDieOffPercent; i++) { //Cull the weaker individuals
			newPopulation[i]=population[i];
		}
		
		int individualIterator = (int) (population.length*populationDieOffPercent);
		int offset = 1; 
		int populationIterator = 0;
		while(individualIterator<populationSize) {
			if(populationIterator==population.length*populationDieOffPercent) {
				offset++;
				individualIterator=0;
			}
			newPopulation[individualIterator] = crossbreedIndividuals(population[populationIterator], population[(populationIterator+offset)%population.length]);
			populationIterator++;
			individualIterator++;
		}
		for(int i = 0; i < population.length; i++) {
			population[i] = newPopulation[i];
		}
		
		
	}
}
