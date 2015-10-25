package training;

import feedforward.FeedForwardNeuralNetwork;

public class MuLambdaEvolution implements Trainer
{
    double[] parameters;
    public MuLambdaEvolution(double[] parameters)
    {
        this.parameters = parameters;
    }

    @Override
    public FeedForwardNeuralNetwork run(FeedForwardNeuralNetwork net, double[][] examples)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
