package training;

import feedforward.FeedForwardNeuralNetwork;

/**
 *
 * @author davej
 */
public class DifferentialEvolution extends GeneticAlgorithm
{

    double[] parameters;
    public DifferentialEvolution(double[] parameters)
    {
        this.parameters = parameters;
    }


    /**
     * This function will use GA to train a neural net
     *
     * @param net
     * @param examples
     * [0] => # of children
     * [1] => # of generations
     * [2] => mutation rate
     * [3] => cross over rate
     * [4] => populationSize
     * [5] => # of weights
     * [6] => crossover Type
     *
     * @return
     */
    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples)
    {
        this.net = net;
        int numbOfChildren = (int) parameters[0];
        int generations = (int) parameters[1];
        double mutationRate = parameters[2];
        double crossoverRate = parameters[3];
        int poopulationSize = (int) parameters[4];
        int numbOfWeights = net.getWeights().length;
        int crossoverType = (int) parameters[6];
        double[][][] paritionedExamples = new double[10][examples.length / 10][];

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < examples.length / 10; j++)
            {
                paritionedExamples[i][j] = examples[i*10 + j];
            }
        }

        double[][] population = initialize(poopulationSize, numbOfWeights);

        for (int i = 0; i < generations; i++)
        {
            //System.out.println("Running Generationg: " + i);
            double[][] children = select(numbOfChildren, population, paritionedExamples[i%10]);
            //System.out.println("Children have been selected");
            children = operate(children, mutationRate, crossoverRate, numbOfChildren, crossoverType);
            //System.out.println("Children have been operated on");
            double[][] totalPop = combine(population, children);
            //System.out.println("Populations have been combined");
            population = Replace(totalPop, poopulationSize, paritionedExamples[i%10]);
            //System.out.println("Replacement has occured");
        }

        //System.out.println("Picking best GA chromosome");
        net.setWeights(getBestWeights(population, examples));

        return net;
    }

}
