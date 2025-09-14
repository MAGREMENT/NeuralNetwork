using System.Windows;

namespace WpfApp.View;

public partial class GraphGuesser
{
    public GraphGuesser()
    {
        InitializeComponent();

        Graph.Points = new Point[]
        {
            new(1, 1),
            new(2, 4)
        };
    }
}