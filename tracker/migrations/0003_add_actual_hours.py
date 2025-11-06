from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tracker', '0002_employeepassword'),
    ]

    operations = [
        migrations.AddField(
            model_name='absencelog',
            name='actual_hours',
            field=models.FloatField(blank=True, null=True, help_text='Actual hours recorded by manager/employee'),
        ),
    ]
