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
     * This method selects individuals from the population to
     * reproduce
     */
    public double[][] select( double[][] currentPop)
    {
        double[][] children = new double[currentPop.length][currentPop[0].length];

        for (int i = 0; i < currentPop.length; i++)
        {
            children[i] = currentPop[i];
        }

        return children;
    }

    /**
     * This method runs the various operators on the Children
     */
    public double[][] operate(double[][] parents, double Beta)
    {
        double[][] modifiedChildren = new double[parents.length][parents[0].length];

        for (int i = 0; i < modifiedChildren.length; i++)
        {
            int x2Index = (int) (Math.random() * parents.length - 1);
            int x3Index = (int) (Math.random() * parents.length - 1);
//            System.out.println("Index 1: " + x2Index + ", Index 2: " + x3Index);
            double[] x2 = parents[x2Index];
            double[] x3 = parents[x3Index];
            double[] ui = mutate(parents[i], x2, x3, Beta);
            modifiedChildren[i] = crossover(parents[i], ui);
        }

        return modifiedChildren;
    }

    /**
     * This method carries out mutation on a child
     */
    public double[] mutate(double[] child, double[] x2, double[] x3, double Beta)
    {
        for (int i = 0; i < child.length; i++)
        {
//            System.out.println(Beta * (x2[i] - x3[i]));
            child[i] = child[i] + Beta * (x2[i] - x3[i]);
        }
        return child;
    }

    /**
     * This method carries out crossover to create a new child
     */
    public double[] crossover(double[] x1, double[] ui)
    {
        double[] child = new double[x1.length];
        int lastPoint = (int) (Math.random() * x1.length);

//        System.out.println("Crossover Point: " + lastPoint + " size: " + x1.length);
        for (int i = 0; i < x1.length; i++)
        {
            if (i < lastPoint)
            {
                child[i] = x1[i];
            }
            else
            {
                child[i] = ui[i];
            }
        }

        return child;
    }

    /**
     * This method uses the parents and children to create a new pool
     * eliminating less fit individuals
     */
    public double[][] Replace(double[][] population, double[][] children, double[][] examples)
    {
        double[][] newPop = new double[population.length][];
        double parentScore;
        double childScore;


        for (int i = 0; i < population.length; i++)
        {
            parentScore = fitnessFunction(population[i], examples);
            childScore = fitnessFunction(children[i], examples);

            if (childScore > parentScore)
            {
                newPop[i] = children[i];
//                System.out.println("using child");
            }
            else
            {
//                System.out.println("using parent");
                newPop[i] = population[i];
            }
        }

        return newPop;
    }



    /**
     * This function will use GA to train a neural net
     *
     * [0] => populationSize
     * [1] => # of generations
     * [2] => Beta
     *
     * @return
     */
    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples)
    {
        this.net = net;
        int populationSize = (int) parameters[0];
        int generations = (int) parameters[1];
        double Beta = parameters[2];
        int numbOfWeights = net.getWeights().length;
        double[][][] paritionedExamples = new double[10][examples.length / 10][];

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < examples.length / 10; j++)
            {
                paritionedExamples[i][j] = examples[i*10 + j];
            }
        }

        double[][] population = initialize(populationSize, numbOfWeights);

        for (int i = 0; i < generations; i++)
        {
            double[][] children = select(population);
            children = operate(children, Beta);
            population = Replace(population, children, paritionedExamples[i%10]);
        }

        net.setWeights(getBestWeights(population, examples));

        return net;
    }

}
