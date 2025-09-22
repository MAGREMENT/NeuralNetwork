using System.Globalization;
using System.Windows;
using Model;
using WpfApp.Presenter;

namespace WpfApp.View;

public partial class GraphGuesser : IGraphGuesserView
{
    private readonly GraphGuesserPresenter _presenter;
    
    public GraphGuesser()
    {
        InitializeComponent();

        _presenter = new GraphGuesserPresenter(this);
        Graph.OutputGetter = _presenter.GetValueFor;
        
        _presenter.Start();
    }

    public void Generate(object? o, RoutedEventArgs args)
    {
        Graph.SetMaxValues(_presenter.Box.UpperX, _presenter.Box.UpperY);
        _presenter.GeneratePoints();
    }

    public void Remove(object? o, RoutedEventArgs args)
    {
        _presenter.RemovePoints();
    }

    public void SetPoints(IReadOnlyList<GuessingPoint> points)
    {
        Graph.Points = points;
    }

    public void SetCost(double v)
    {
        CostBlock.Text = v.ToString("0.00", CultureInfo.CreateSpecificCulture("en-US"));
    }

    private void Learn(object sender, RoutedEventArgs e)
    {
        _presenter.Learn();
        Graph.Refresh();
    }
}