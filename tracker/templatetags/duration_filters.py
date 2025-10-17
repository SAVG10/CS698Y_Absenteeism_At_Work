from django import template
import math

register = template.Library()

@register.filter
def hours_to_duration(total_hours):
    """
    Converts a float number of hours (e.g., 8.52)
    into a string (e.g., "8h 31m").
    """
    if total_hours is None:
        return "0h 0m"

    # math.modf splits the float into its fractional and integer parts
    # e.g., 8.52 -> (0.52, 8.0)
    frac_hours, int_hours = math.modf(float(total_hours))

    # Convert the fractional hour (0.52) into whole minutes
    total_minutes = int(frac_hours * 60)

    # Return the formatted string
    return f"{int(int_hours)}h {total_minutes}m"