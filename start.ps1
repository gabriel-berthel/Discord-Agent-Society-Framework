# Start each Python server with the respective arguments
Start-Process python -ArgumentList "server.py 1.env bot_config.yaml fact_checker"
Start-Process python -ArgumentList "server.py 2.env bot_config.yaml baseline"
Start-Process python -ArgumentList "server.py 3.env bot_config.yaml activist"
Start-Process python -ArgumentList "server.py 4.env bot_config.yaml trouble_maker"
Start-Process python -ArgumentList "server.py 5.env bot_config.yaml mediator"
