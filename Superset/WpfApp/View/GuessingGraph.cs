using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Model;

namespace WpfApp.View;

public class GuessingGraph : FrameworkElement
{
    private IReadOnlyList<GuessingPoint> _points = Array.Empty<GuessingPoint>();
    private double _scaleX = 10;
    private double _scaleY = 10;

    public IReadOnlyList<GuessingPoint> Points
    {
        set
        {
            _points = value;
            Refresh();
        }
    }
    
    public Func<double, double, int>? OutputGetter { get; set; }
    
    // Provide a required override for the VisualChildrenCount property.
    protected override int VisualChildrenCount => 0;

    // Provide a required override for the GetVisualChild method.
    protected override Visual GetVisualChild(int index)
    {
        return null!;
    }
    
    public void Refresh()
    {
        Dispatcher.Invoke(InvalidateVisual);
    }

    public void SetMaxValues(double x, double y)
    {
        _scaleX = ActualWidth / x;
        _scaleY = ActualHeight / y;
    }
    
    protected override void OnRender(DrawingContext context)
    {
        if (double.IsNaN(ActualWidth) || ActualWidth == 0 ||
            double.IsNaN(ActualHeight) || ActualWidth == 0) return;

        if (OutputGetter is not null)
        {
            const double space = 4;
            for (double x = 0; x < ActualWidth; x += space)
            {
                for (double y = 0; y < ActualHeight; y += space)
                {
                    var o = OutputGetter(x, y);
                    context.DrawRectangle(GetBrushBackground(o), null, new Rect(x, y, space, space));
                }
            }
        }

        const double offset = 5;
        const double width = 5;
        
        context.DrawRectangle(Brushes.Black, null, new Rect(offset, offset, width, ActualHeight - offset * 2));
        context.DrawRectangle(Brushes.Black, null, new Rect(offset, ActualHeight - offset - width, 
            ActualWidth - offset * 2, width));

        const double radius = 3;
        
        foreach (var point in _points)
        {
            var p = new Point(point.X  * _scaleX + offset + width, ActualHeight - point.Y * _scaleY - offset - width);
            context.DrawEllipse(GetBrush(point.Output), null, p, radius, radius);
        }
    }

    private static Brush GetBrush(int v)
    {
        return v switch
        {
            0 => Brushes.Blue,
            1 => Brushes.Red,
            _ => Brushes.Black
        };
    }
    
    private static Brush GetBrushBackground(int v)
    {
        return v switch
        {
            0 => Brushes.Aqua,
            1 => Brushes.Tomato,
            _ => Brushes.Black
        };
    }
}