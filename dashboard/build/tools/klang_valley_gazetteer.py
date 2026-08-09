# Klang Valley gazetteer: (name, lat, lon, district)
GAZETTEER = [
    # Kuala Lumpur core
    ("KLCC", 3.1578, 101.7123), ("Bukit Bintang", 3.1466, 101.7100),
    ("Chow Kit", 3.1637, 101.6969), ("Titiwangsa", 3.1745, 101.7009),
    ("Setapak", 3.1970, 101.7186), ("Wangsa Maju", 3.2044, 101.7336),
    ("Kepong", 3.2148, 101.6343), ("Segambut", 3.1877, 101.6690),
    ("Sentul", 3.1780, 101.6910), ("Brickfields", 3.1290, 101.6860),
    ("Bangsar", 3.1290, 101.6710), ("Bangsar South", 3.1080, 101.6620),
    ("Mid Valley", 3.1180, 101.6780), ("Cheras", 3.1010, 101.7440),
    ("Sri Petaling", 3.0645, 101.6870), ("Bukit Jalil", 3.0570, 101.6820),
    ("Old Klang Road", 3.0710, 101.6650), ("Salak Selatan", 3.0850, 101.7080),
    ("Sungai Besi", 3.0650, 101.7090), ("Pantai Dalam", 3.1020, 101.6660),
    ("Kuchai Lama", 3.0850, 101.6870), ("Taman Tun Dr Ismail", 3.1520, 101.6280),
    ("Damansara Heights", 3.1470, 101.6620), ("Mont Kiara", 3.1720, 101.6510),
    ("Sri Hartamas", 3.1650, 101.6520), ("Jinjang", 3.2140, 101.6570),
    ("Melawati", 3.2050, 101.7440), ("Ampang", 3.1490, 101.7620),
    ("Ampang Jaya", 3.1450, 101.7690), ("Pandan Indah", 3.1330, 101.7370),
    ("Pandan Jaya", 3.1300, 101.7290), ("Setiawangsa", 3.1780, 101.7280),
    ("Keramat", 3.1620, 101.7290), ("Maluri", 3.1290, 101.7220),
    ("Cheras Hartamas", 3.0930, 101.7590), ("Taman Midah", 3.0960, 101.7460),
    ("Bandar Tun Razak", 3.0800, 101.7220), ("Bukit Bintang City Centre", 3.1420, 101.7150),

    # Putrajaya / Cyberjaya / Sepang
    ("Putrajaya", 2.9264, 101.6964), ("Cyberjaya", 2.9220, 101.6540),
    ("Sepang", 2.7297, 101.7420), ("Salak Tinggi", 2.8060, 101.6890),
    ("KLIA", 2.7456, 101.7099), ("KLIA2", 2.7590, 101.6960),
    ("Dengkil", 2.8390, 101.6520), ("Bandar Baru Salak Tinggi", 2.8130, 101.6870),
    ("Nilai Impian", 2.8330, 101.7690), ("Bandar Bukit Mahkota", 2.8880, 101.7500),

    # Gombak district
    ("Gombak", 3.2280, 101.7100), ("Batu Caves", 3.2379, 101.6840),
    ("Selayang", 3.2500, 101.6480), ("Rawang", 3.3190, 101.5760),
    ("Sungai Buloh", 3.2000, 101.5700), ("Hulu Kelang", 3.1980, 101.7460),
    ("Taman Melawati", 3.2130, 101.7460), ("Batu Arang", 3.3750, 101.4570),
    ("Serendah", 3.3660, 101.5980), ("Kuang", 3.2960, 101.5540),
    ("Bukit Antarabangsa", 3.1830, 101.7620), ("Zoo Negara Area", 3.2150, 101.7590),
    ("Sri Gombak", 3.2600, 101.6990), ("Templer's Park", 3.2900, 101.6280),

    # Petaling district
    ("Petaling Jaya", 3.1073, 101.6067), ("PJ Old Town", 3.1040, 101.6420),
    ("Kelana Jaya", 3.0850, 101.5940), ("Damansara Utama", 3.1350, 101.6220),
    ("Damansara Jaya", 3.1370, 101.6280), ("Bandar Utama", 3.1490, 101.6070),
    ("Sunway", 3.0670, 101.6060), ("Subang Jaya", 3.0570, 101.5850),
    ("USJ", 3.0430, 101.5820), ("Puchong", 3.0356, 101.6172),
    ("Bukit Puchong", 2.9970, 101.6060), ("Seri Kembangan", 3.0290, 101.7020),
    ("Serdang", 2.9930, 101.7200), ("Taman Equine", 2.9800, 101.6810),
    ("Kota Damansara", 3.1600, 101.5850), ("Damansara Perdana", 3.1660, 101.5960),
    ("Sri Damansara", 3.1970, 101.6320), ("Subang", 3.1080, 101.5480),
    ("Ara Damansara", 3.1180, 101.5820), ("Glenmarie", 3.0940, 101.5680),
    ("Bandar Sunway", 3.0680, 101.6020), ("Puchong Perdana", 3.0170, 101.6070),
    ("Puchong Utama", 3.0090, 101.5920), ("IOI Puchong", 3.0470, 101.6260),

    # Klang district
    ("Klang", 3.0333, 101.4500), ("Port Klang", 3.0000, 101.4000),
    ("Kapar", 3.1330, 101.4210), ("Meru", 3.0940, 101.4460),
    ("Bukit Tinggi Klang", 3.0300, 101.4630), ("Kota Kemuning", 2.9900, 101.5330),
    ("Bukit Rimau", 2.9980, 101.5210), ("Shah Alam", 3.0733, 101.5185),
    ("i-City Shah Alam", 3.0620, 101.5470), ("Setia Alam", 3.1100, 101.4650),
    ("Bandar Bukit Raja", 3.0670, 101.4400), ("Klang Utara", 3.0550, 101.4700),
    ("Pandamaran", 2.9980, 101.4180), ("Telok Gong", 2.9740, 101.3820),
    ("Bukit Kapar", 3.1470, 101.3810), ("Section 7 Shah Alam", 3.0680, 101.5310),
    ("Elmina", 3.1770, 101.4700), ("Denai Alam", 3.1310, 101.4900),

    # Hulu Langat district
    ("Kajang", 2.9930, 101.7870), ("Bangi", 2.9200, 101.7830),
    ("Semenyih", 2.9490, 101.8460), ("Batu 9 Cheras", 3.0350, 101.7570),
    ("Balakong", 3.0290, 101.7460), ("Beranang", 2.8940, 101.8720),
    ("Kajang Utama", 2.9970, 101.7900), ("Sungai Chua", 2.9660, 101.7660),
    ("Bandar Baru Bangi", 2.9440, 101.8060), ("Batu 14 Hulu Langat", 3.0700, 101.8280),
    ("Hulu Langat Town", 3.0900, 101.8380), ("Cheras Batu 11", 3.0480, 101.7580),

    # Kuala Langat district
    ("Banting", 2.8130, 101.5030), ("Morib", 2.7500, 101.4610),
    ("Telok Panglima Garang", 2.9330, 101.4700), ("Jenjarom", 2.8960, 101.4520),
    ("Tanjung Sepat", 2.6650, 101.4310), ("Kelanang", 2.8300, 101.4700),
    ("Bandar Saujana Putra", 2.9550, 101.5750), ("Bukit Changgang", 2.8580, 101.5700),
    ("Sijangkang", 2.9410, 101.5390), ("Klang Selatan", 2.9700, 101.4900),
]

def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (0-360, 0=N) from point 1 to point 2."""
    from math import radians, degrees, sin, cos, atan2
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(lat2r)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360


COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def nearest_place(lat, lon):
    best, bestd = None, float("inf")
    for name, glat, glon in GAZETTEER:
        d = haversine(lat, lon, glat, glon)
        if d < bestd:
            bestd, best = d, name
    return best, bestd


def nearest_place_label(lat, lon, core_radius_km=0.25):
    """Human label that differentiates hexes sharing the same nearest
    gazetteer landmark: 'Petaling Jaya' for the hex right at the landmark,
    'Petaling Jaya (NE, 1.2km)' for one 1.2km northeast of it, etc.
    Falls back to the bare landmark name when the hex centroid IS the
    landmark's own cell.
    """
    best, bestd, blat, blon = None, float("inf"), None, None
    for name, glat, glon in GAZETTEER:
        d = haversine(lat, lon, glat, glon)
        if d < bestd:
            bestd, best, blat, blon = d, name, glat, glon
    if bestd < core_radius_km:
        return best, bestd
    brng = bearing(blat, blon, lat, lon)
    idx = round(brng / 45) % 8
    direction = COMPASS[idx]
    dist_txt = f"{int(round(bestd * 1000))}m" if bestd < 1 else f"{bestd:.1f}km"
    return f"{best} ({direction}, {dist_txt})", bestd
