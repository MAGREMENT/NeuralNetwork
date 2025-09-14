using System.Windows;
using System.Windows.Media;

namespace WpfApp.View;

public class Graph : FrameworkElement
{
    private IReadOnlyList<Point> _points = Array.Empty<Point>();

    public IReadOnlyList<Point> Points
    {
        set
        {
            _points = value;
            Refresh();
        }
    }
    
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
    
    protected override void OnRender(DrawingContext context)
    {
        if (double.IsNaN(ActualWidth) || ActualWidth == 0 ||
            double.IsNaN(ActualHeight) || ActualWidth == 0) return;

        const double offset = 5;
        const double width = 5;
        
        context.DrawRectangle(Brushes.Black, null, new Rect(offset, offset, width, ActualHeight - offset * 2));
        context.DrawRectangle(Brushes.Black, null, new Rect(offset, ActualHeight - offset - width, 
            ActualWidth - offset * 2, width));

        const double radius = 3;
        const double scale = 10;
        
        foreach (var point in _points)
        {
            var p = new Point(point.X  * scale + offset + width, ActualHeight - point.Y * scale - offset - width);
            context.DrawEllipse(Brushes.Black, null, p, radius, radius);
        }
    }
}