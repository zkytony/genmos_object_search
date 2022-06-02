obj_id_map = {
    "G": 12,  # Toyota
    "B": 23,  # Bike
    "R": 34   # honda
}
abrv_letter_map = {
    "gcar": "G",
    "bike": "B",
    "rcar": "R"
}
symbol_letter_map = {
    "GreenToyota": "G",
    "RedBike": "B",
    "RedHonda": "R",
}
obj_letter_map = {**abrv_letter_map,
                  **symbol_letter_map}

letter_symbol_map = {symbol_letter_map[s]:s
                     for s in symbol_letter_map}
