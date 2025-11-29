using Base;

namespace WpfApp.Presenter;

public interface IPresenterService
{
    public static IPresenterService Instance { get; } = new V1PresenterService();
    
    public INeuralNetwork GetDoodleGuesserNetwork();
    public (INeuralNetwork, ILearningState) GetGraphGuesserNetwork();
}