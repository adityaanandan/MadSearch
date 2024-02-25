from django.shortcuts import render
from core.listy import *

# Create your views here.
def index(request): 
    if request.method == 'POST': 
        budget = int(request.POST["budget"])
        distance = int(request.POST["distance"])
        bedrooms = int(request.POST["bedrooms"])
        print(budget)
        main_df = app(budget, bedrooms, distance)
        columns = ['address', 'sq ft', 'Unit Type', 'Available', 'Bedroom(s)', 'Bathroom(s)', 'Link']
    
        df = main_df[columns]

        context = {}

        context["datatable"] = df.to_dict(orient="tight", index=True)

        return render(request, 'table.html', context)

    return render(request, 'index.html')

