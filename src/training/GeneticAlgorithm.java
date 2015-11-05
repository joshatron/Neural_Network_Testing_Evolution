package training;

import com.sun.tools.javac.jvm.Gen;
import feedforward.FeedForwardNeuralNetwork;

public class GeneticAlgorithm implements Trainer
{
    double[] parameters;
    FeedForwardNeuralNetwork net;

    public GeneticAlgorithm()
    {
        System.out.println("GA default constructor called");
    }

    public GeneticAlgorithm(double[] parameters)
    {
        this.parameters = parameters;
    }

    /**
     * This method initializes the GA population using random values
     */
    public double[][] initialize(int populationSize, int numbOfWeights)
    {
        double[][] population = new double[populationSize][numbOfWeights];

        for (int i = 0; i < populationSize; i++)
        {
            this.net.generateRandomWeights();
            population[i] = this.net.getWeights();
//            for (int j = 0; j < numbOfWeights; j++)
//            {
//                population[i][j] = Math.random() * 2 - 1;
//            }
        }

        return population;
    }

    /**
     * This method selects individuals from the population to
     * reproduce
     */
    public double[][] select(int populationSize, double[][] currentPop, double[][] examples)
    {
        double[] values = new double[currentPop.length];
        double[][] children = new double[populationSize][currentPop[0].length];

        for (int i = 0; i < currentPop.length; i++)
        {
            values[i] = fitnessFunction(currentPop[i], examples);
        }

        int numbOfChildren = 0;
        int currentParent = 0;

        while (numbOfChildren < populationSize)
        {
            if (Math.random() < values[currentParent])
            {
                children[numbOfChildren] = currentPop[currentParent];
                numbOfChildren++;
            }

            currentParent = (currentParent + 1) % currentPop.length;
        }

        return children;
    }

    /**
     * This method runs the various operators on the Children
     */
    public double[][] operate(double[][] parents, double mutationRate, double crossoverRate, int NumbOfParents, int crossoverType)
    {
        double[][] modifiedChildren = new double[parents.length][parents[0].length];

        for (int i = 0; i < modifiedChildren.length; i++)
        {
            modifiedChildren[i] = parents[i];
            if (Math.random() < crossoverRate)
            {
                double[][] crossoverParents = new double[NumbOfParents][modifiedChildren[i].length];
                crossoverParents[0] = modifiedChildren[i];

                for (int j = 1; j < crossoverParents.length; j++)
                {
                    int nextParent = (int) (Math.random() * parents.length);
                    crossoverParents[j] = parents[nextParent];
                }

                modifiedChildren[i] = crossover(crossoverParents, crossoverType);
            }
            modifiedChildren[i] = mutate(modifiedChildren[i], mutationRate);
        }

        return modifiedChildren;
    }

    /**
     * This method carries out mutation on a child
     */
    public double[] mutate(double[] child, double rate)
    {
        for (int i = 0; i < child.length; i++)
        {
            if (Math.random() < rate)
            {
                child[i] += (Math.random() * 2) - 1;
            }
        }
        return child;
    }

    /**
     * This method carries out crossover to create a new child
     */
    public double[] crossover(double[][] parents, int crossoverType)
    {
        double[] child = new double[parents[0].length];
        // swap crossover
        if (crossoverType == 0)
        {
            int[] crossoverPoints = new int[parents.length - 1];
            int lastPoint = 0;
            for (int i = 0; i < parents.length - 1; i++)
            {
                crossoverPoints[i] = lastPoint + (int) (Math.random() * (parents[0].length - lastPoint));
                lastPoint = crossoverPoints[i];
            }

            for (int j = 0; j < parents[0].length; j++)
            {
                for (int k = 0; k < crossoverPoints.length; k++)
                {
                    if (j < crossoverPoints[k])
                    {
                        child[j] = parents[k][j];
                        k = crossoverPoints.length + 1;
                    }
                }
            }

        }
        // point by point crossover
        else if (crossoverType == 1)
        {
            for (int i = 0; i < parents[0].length; i++)
            {
                int parent = (int) (Math.random() * parents.length);

                child[i] = parents[parent][i];
            }
        }

        return child;
    }

    /**
     * This method uses the parents and children to create a new pool
     * eliminating less fit individuals
     */
    public double[][] Replace(double[][] population, int size, double[][] examples)
    {
        double[][] newPop = new double[size][population[0].length];
        int currentSize = 0;
        int index = 0;
        double[] scores = new double[population.length];

        for (int i = 0; i < population.length; i++)
        {
            scores[i] = fitnessFunction(population[i], examples);
        }

        while (currentSize < size)
        {
            if (Math.random() < scores[index])
            {
                newPop[currentSize] = population[index];
                currentSize++;
            }

            index = (index + 1) % scores.length;
        }

        return newPop;
    }

    /**
     * This method evaluates the fitness of an individual
     */
    public double fitnessFunction(double[] weights, double[][] examples)
    {
        double score = 0;
        net.setWeights(weights);
        int trials = 0;
        for (int i = 0; i < Math.min(examples.length, 100); i++)
        {
            trials++;
            double[] confidences = this.net.compute(examples[i]);
            int len = examples[i].length - 1;

            int index = 0;
            double bestConfidence = 0;

            for (int j = 0; j < confidences.length; j++)
            {
                if (confidences[j] > bestConfidence)
                {
                    bestConfidence = confidences[j];
                    index = j;
                }
                else
                {
                    //System.out.println(confidences[j]);
                }
            }

           // System.out.println(bestConfidence);

            //System.out.println(examples[i][len]);
            if (index == examples[i][len])
            {
//                System.out.println("We were correct!");
                //score += bestConfidence;
                score += 1;
            }
            else if (bestConfidence > 0)
            {
//                System.out.println("%%%%%%%%%%%%%%%%%%%%%");
                score += Math.exp(-1 * bestConfidence);
            }
        }

//        if (score < (trials / 10)) {
//            score = trials / 10;
//        }

        //System.out.println(score);

        return score / (trials);
    }

    /**
     * This method combines two arrays
     */
    public double[][] combine(double[][] array1, double[][] array2)
    {
        int size = array1.length + array2.length;

        double[][] combo = new double[size][array1[0].length];

        for (int i = 0; i < combo.length; i++)
        {
            if (i < array1.length)
            {
                combo[i] = array1[i];
            }
            else
            {
                combo[i] = array2[i-array1.length];
            }
        }

        return combo;
    }

    public double[] getBestWeights(double[][] pop, double[][] examples)
    {
        double[] best = pop[1];
        double bestFitness = fitnessFunction(best, examples);

        for (int i = 1; i < pop.length; i++)
        {
            double[] current = pop[1];
            double currentFitness = fitnessFunction(current, examples);
            if (currentFitness > bestFitness)
            {
                bestFitness = currentFitness;
                best = current;
            }
        }

        return best;
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

//
//    public static void main(String[] args)
//    {
//        GeneticAlgorithm ga = new GeneticAlgorithm();
//        FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork();
//        ga.run()
//    }

}
