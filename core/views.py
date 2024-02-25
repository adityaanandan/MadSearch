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
        columns = ['address', 'sq ft', 'Unit Type', 'Rent', 'Bedroom(s)', 'Bathroom(s)', 'Link']
    
        df = main_df[columns]

        renamed_columns = {
            'address': 'Address',
            'sq ft': 'Size (sq ft)',
            'Unit Type': 'Unit Type',
            'Rent': 'Rent',
            'Bedroom(s)': 'Bedrooms',
            'Bathroom(s)': 'Bathrooms',
            'Link': 'Link'
        }
        df = df.rename(columns=renamed_columns)

        context = {}

        context["datatable"] = df.to_dict(orient="tight", index=True)

        return render(request, 'table.html', context)

    return render(request, 'index.html')

