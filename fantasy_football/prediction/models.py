from django.db import models

# example model
from django.db import models

class Player(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    position = models.CharField(max_length=50)
    team = models.CharField(max_length=50)
    # Add more fields as needed

    def __str__(self):
        return self.name
