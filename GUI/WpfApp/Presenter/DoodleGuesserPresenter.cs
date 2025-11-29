using Base;
using Model;

namespace WpfApp.Presenter;

using NeuralNetwork.V1;

public class DoodleGuesserPresenter
{
    private readonly IDoodleGuesserView _view;
    
    private readonly Doodle _doodle = new(28, 28);
    private readonly IReadOnlyList<GuessingPoint> _dataSet = MNIST.Read(
        "mnist-data/t10k-labels.idx1-ubyte", 
        "mnist-data/t10k-images.idx3-ubyte", 10000);

    private readonly Random _random = new();

    private readonly INeuralNetwork _network;
    private int _index = -1;

    public DoodleGuesserPresenter(IDoodleGuesserView view)
    {
        _view = view;
        _network = IPresenterService.Instance.GetDoodleGuesserNetwork();
    }

    public void Next()
    {
        _index = _random.Next(10000);
        
        _doodle.SetData(_dataSet[_index].Values);
        _view.SetDoodleData(_doodle.To2DData());
        _view.SetExpected(_dataSet[_index].Output);
        Predict();
    }

    public void Clear()
    {
        _index = -1;
        _doodle.Clear();
        _view.SetDoodleData(_doodle.To2DData());
        Predict();
    }

    public void Draw(int col, int row, double colPercent, double rowPercent)
    {
        _doodle.Draw(col, row, colPercent, rowPercent);
        _view.SetDoodleData(_doodle.To2DData());
    }

    public void Predict()
    {
        var output = _network.Predict(_doodle.ToNeuralNetworkInputs());

        var predictions = new List<(int, double)>(10);
        for (int i = 0; i < output.Length; i++)
        {
            predictions.Add((i, Math.Round(output[i] * 100, 2)));
        }
        
        predictions.Sort((a, b) => b.Item2.CompareTo(a.Item2));
        _view.SetPredictions(predictions);
    }
}

public interface IDoodleGuesserView
{
    void SetDoodleData(double[,] data);
    void SetPredictions(IReadOnlyList<(int, double)> predictions);
    void SetExpected(int n);
}