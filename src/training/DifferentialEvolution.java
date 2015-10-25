package training;

import feedforward.FeedForwardNeuralNetwork;

/**
 *
 * @author davej
 */
public class DifferentialEvolution implements Trainer
{

    double[] parameters;
    public DifferentialEvolution(double[] parameters)
    {
        this.parameters = parameters;
    }
    
    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
