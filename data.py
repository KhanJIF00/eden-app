import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate comprehensive aquaculture dataset (12,000 records)
# Simulate data from multiple aquaculture operations over 3 years

print("Generating Aquaculture Water Quality Dataset...")
print("=" * 55)

# Time range: 3 years with multiple daily measurements
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)
base_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create realistic sampling schedule
dates = []
farm_ids = []
species_type = []
system_type = []

# Define farm types and species
farm_types = [
    ('pond', 'tilapia'), ('pond', 'catfish'), ('pond', 'carp'),
    ('raceway', 'trout'), ('raceway', 'salmon'), 
    ('tank', 'bass'), ('tank', 'shrimp'), ('cage', 'salmon'),
    ('pond', 'hybrid_striped_bass'), ('recirculating', 'tilapia')
]

farm_names = [f"Farm_{i:02d}" for i in range(1, 21)]  # 20 different farms

for date in base_dates:
    # More frequent sampling during growing season (spring/summer)
    season_factor = 1 + 0.5 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365)
    n_samples = max(1, int(np.random.poisson(2 * season_factor)))
    
    for _ in range(n_samples):
        # Random sampling time during daylight hours
        sample_time = date + timedelta(hours=random.randint(6, 18), 
                                     minutes=random.randint(0, 59))
        dates.append(sample_time)
        
        # Assign farm and system characteristics
        farm_id = random.choice(farm_names)
        system, species = random.choice(farm_types)
        
        farm_ids.append(farm_id)
        system_type.append(system)
        species_type.append(species)

# Trim to exactly 12,000 records
target_size = 12000
if len(dates) > target_size:
    indices = sorted(random.sample(range(len(dates)), target_size))
    dates = [dates[i] for i in indices]
    farm_ids = [farm_ids[i] for i in indices]
    system_type = [system_type[i] for i in indices]
    species_type = [species_type[i] for i in indices]

n_points = len(dates)
print(f"Generating {n_points:,} data points for aquaculture systems...")

# Research-based parameter ranges for aquaculture
OPTIMAL_RANGES = {
    'temperature': {
        'coldwater': (10, 18),    # Trout, salmon
        'coolwater': (15, 25),    # Bass, hybrid striped bass  
        'warmwater': (20, 30),    # Tilapia, catfish, carp
        'tropical': (24, 32)      # Shrimp
    },
    'ph': (6.0, 9.0),           # Research: 6-9 optimal, <4.5 or >10 lethal
    'dissolved_oxygen': {
        'minimum': 5,            # Research: 5 mg/L warmwater, 7 mg/L coldwater
        'optimal': (6, 12),
        'critical': 2            # <2 mg/L lethal, <3 mg/L for some species
    },
    'ammonia_nh4': {
        'safe': (0.02, 0.05),   # Research: 0.02-0.05 mg/L safe
        'stress': (0.05, 0.5),   # Causes stress
        'toxic': 0.5            # >0.5 mg/L toxic
    },
    'nitrate': {
        'optimal': (0.1, 4.0),   # Research: 0.1-4.0 mg/L favorable
        'safe': (0, 90),         # <90 mg/L not toxic
        'typical': (0, 25)       # Typical range in well-managed systems
    },
    'conductivity': {
        'freshwater': (50, 1500),  # μS/cm
        'brackish': (1500, 5000),
        'marine': (35000, 55000)
    },
    'turbidity': {
        'clear': (1, 10),        # Clear water systems
        'moderate': (10, 40),    # Some turbidity acceptable
        'high': (40, 100)        # High turbidity from algae/particles
    }
}

def get_species_temp_category(species):
    """Get temperature category based on species"""
    if species in ['trout', 'salmon']:
        return 'coldwater'
    elif species in ['bass', 'hybrid_striped_bass']:
        return 'coolwater'
    elif species == 'shrimp':
        return 'tropical'
    else:  # tilapia, catfish, carp
        return 'warmwater'

def get_system_conductivity_range(system, species):
    """Get conductivity range based on system type"""
    if species == 'shrimp':
        return OPTIMAL_RANGES['conductivity']['brackish']
    elif system in ['cage'] and random.random() < 0.3:  # Some marine cages
        return OPTIMAL_RANGES['conductivity']['marine']
    else:
        return OPTIMAL_RANGES['conductivity']['freshwater']

# Generate temperature based on species requirements and seasonality
def generate_temperature(dates, species_list, system_list):
    temperatures = []
    for i, date in enumerate(dates):
        species = species_list[i]
        system = system_list[i]
        
        # Get optimal range for species
        temp_category = get_species_temp_category(species)
        optimal_min, optimal_max = OPTIMAL_RANGES['temperature'][temp_category]
        optimal_temp = (optimal_min + optimal_max) / 2
        
        # Seasonal variation
        day_of_year = date.timetuple().tm_yday
        seasonal_amplitude = 8 if system == 'pond' else 4  # Ponds have more variation
        seasonal_temp = optimal_temp + seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # System-specific modifications
        if system == 'recirculating':
            # More controlled temperature
            temp = seasonal_temp * 0.7 + optimal_temp * 0.3
        elif system in ['tank', 'raceway']:
            # Some temperature control
            temp = seasonal_temp * 0.8 + optimal_temp * 0.2
        else:  # pond, cage - more environmental variation
            temp = seasonal_temp
        
        # Add daily variation and measurement noise
        daily_var = np.random.normal(0, 1.5)
        final_temp = temp + daily_var
        
        # Ensure within reasonable bounds
        final_temp = max(optimal_min - 5, min(optimal_max + 8, final_temp))
        temperatures.append(final_temp)
    
    return temperatures

# Generate pH based on system type and management practices
def generate_ph(dates, system_list, species_list):
    ph_values = []
    for i, date in enumerate(dates):
        system = system_list[i]
        species = species_list[i]
        
        # Base pH varies by system
        if system == 'recirculating':
            base_ph = 7.2 + random.uniform(-0.3, 0.5)  # Well buffered
        elif system in ['tank', 'raceway']:
            base_ph = 7.5 + random.uniform(-0.5, 0.8)  # Moderate control
        else:  # pond, cage
            base_ph = 7.8 + random.uniform(-1.0, 1.2)  # More variable
        
        # Daily pH fluctuations from photosynthesis (more in ponds)
        hour = date.hour
        if system == 'pond':
            # pH rises during day (photosynthesis), drops at night
            daily_cycle = 0.3 * np.sin(2 * np.pi * (hour - 6) / 24) if 6 <= hour <= 18 else -0.2
        else:
            daily_cycle = 0.1 * np.sin(2 * np.pi * (hour - 12) / 24)
        
        ph_val = base_ph + daily_cycle + np.random.normal(0, 0.15)
        
        # Ensure within viable range
        ph_val = max(5.5, min(9.5, ph_val))
        ph_values.append(ph_val)
    
    return ph_values

# Generate dissolved oxygen with realistic daily cycles and system effects
def generate_dissolved_oxygen(dates, temperatures, system_list, species_list):
    do_values = []
    for i, date in enumerate(dates):
        temp = temperatures[i]
        system = system_list[i]
        species = species_list[i]
        
        # Temperature-dependent saturation (warmer water holds less O2)
        do_saturation = 14.6 - 0.41 * temp + 0.008 * temp**2  # Empirical formula
        
        # System-specific DO management
        if system == 'recirculating':
            target_do = do_saturation * 0.9  # Well aerated
        elif system in ['raceway', 'tank']:
            target_do = do_saturation * 0.8  # Good aeration
        elif system == 'cage':
            target_do = do_saturation * 0.7  # Natural water movement
        else:  # pond
            target_do = do_saturation * 0.6  # Most variable
        
        # Daily cycle (higher during day due to photosynthesis in ponds)
        hour = date.hour
        if system == 'pond':
            daily_cycle_amplitude = 2.0
        else:
            daily_cycle_amplitude = 0.5
        
        daily_cycle = daily_cycle_amplitude * np.sin(2 * np.pi * (hour - 6) / 24) if 6 <= hour <= 20 else -0.5
        
        # Stocking density effect (higher density = lower DO)
        density_effect = random.uniform(-1, 0) if random.random() < 0.3 else 0
        
        do_val = target_do + daily_cycle + density_effect + np.random.normal(0, 0.8)
        
        # Species-specific minimum requirements
        species_min = 7 if species in ['trout', 'salmon'] else 5
        do_val = max(1.0, do_val)  # Absolute minimum for data realism
        
        do_values.append(do_val)
    
    return do_values

# Generate ammonia (NH4) based on feeding, bioload, and system efficiency
def generate_ammonia_nh4(dates, system_list, species_list, temperatures):
    nh4_values = []
    for i, date in enumerate(dates):
        system = system_list[i]
        species = species_list[i]
        temp = temperatures[i]
        
        # Base ammonia production varies by system efficiency
        if system == 'recirculating':
            base_nh4 = 0.02  # Excellent biofilter
        elif system in ['raceway', 'tank']:
            base_nh4 = 0.08  # Good water exchange
        elif system == 'cage':
            base_nh4 = 0.05  # Natural dilution
        else:  # pond
            base_nh4 = 0.15  # Most variable, depends on management
        
        # Temperature effect (higher temp = more bacterial activity and ammonia production)
        temp_factor = 1 + (temp - 20) * 0.02
        
        # Feeding cycle effect (higher ammonia 2-6 hours after feeding)
        hour = date.hour
        if 10 <= hour <= 16:  # Post-feeding peak
            feeding_effect = random.uniform(1.2, 2.5)
        else:
            feeding_effect = random.uniform(0.8, 1.2)
        
        # Seasonal bioload effect
        day_of_year = date.timetuple().tm_yday
        seasonal_bioload = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 100) / 365)  # Peak in summer
        
        # Occasional management issues
        management_spike = 1.0
        if random.random() < 0.05:  # 5% chance of elevated ammonia
            management_spike = random.uniform(2, 8)
        
        nh4_val = base_nh4 * temp_factor * feeding_effect * seasonal_bioload * management_spike
        nh4_val = max(0.001, nh4_val + np.random.lognormal(0, 0.5))
        
        # Cap at reasonable maximum
        nh4_val = min(2.0, nh4_val)
        
        nh4_values.append(nh4_val)
    
    return nh4_values

# Generate nitrate based on nitrification process and system maturity
def generate_nitrate(dates, nh4_values, system_list):
    nitrate_values = []
    system_ages = {farm: random.randint(30, 1000) for farm in set(farm_ids)}  # Days since startup
    
    for i, date in enumerate(dates):
        nh4 = nh4_values[i]
        system = system_list[i]
        farm = farm_ids[i]
        
        # System maturity affects nitrification efficiency
        system_age = system_ages[farm]
        maturity_factor = min(1.0, system_age / 90)  # Mature after ~90 days
        
        # Base nitrate from nitrification
        if system == 'recirculating':
            conversion_rate = 0.8 * maturity_factor  # Efficient biofilter
        elif system in ['tank', 'raceway']:
            conversion_rate = 0.4 * maturity_factor
        elif system == 'pond':
            conversion_rate = 0.6 * maturity_factor  # Natural bacteria
        else:  # cage
            conversion_rate = 0.1  # Open system, dilution
        
        # Convert ammonia to nitrate (simplified)
        converted_nitrate = nh4 * conversion_rate * 4.4  # Molecular weight conversion
        
        # Background nitrate accumulation
        if system == 'recirculating':
            background_nitrate = random.uniform(10, 80)
        elif system in ['tank', 'raceway']:
            background_nitrate = random.uniform(2, 25)
        elif system == 'pond':
            background_nitrate = random.uniform(0.5, 15)
        else:  # cage
            background_nitrate = random.uniform(0.1, 5)
        
        total_nitrate = converted_nitrate + background_nitrate
        
        # Water changes reduce nitrate
        if random.random() < 0.1:  # 10% chance of recent water change
            total_nitrate *= random.uniform(0.3, 0.7)
        
        total_nitrate = max(0.05, total_nitrate + np.random.normal(0, total_nitrate * 0.2))
        nitrate_values.append(min(150, total_nitrate))  # Cap at reasonable max
    
    return nitrate_values

# Generate conductivity based on system type and water source
def generate_conductivity(system_list, species_list):
    conductivities = []
    for i in range(len(system_list)):
        system = system_list[i]
        species = species_list[i]
        
        # Get appropriate range
        cond_min, cond_max = get_system_conductivity_range(system, species)
        
        if species == 'shrimp':
            # Brackish water systems
            base_conductivity = random.uniform(2000, 4500)
        elif system == 'cage' and random.random() < 0.2:
            # Some marine cage operations
            base_conductivity = random.uniform(40000, 52000)
        else:
            # Freshwater systems
            if system == 'recirculating':
                base_conductivity = random.uniform(200, 800)  # More controlled
            elif system in ['tank', 'raceway']:
                base_conductivity = random.uniform(150, 1200)
            else:  # pond
                base_conductivity = random.uniform(100, 2000)  # Most variable
        
        # Add measurement variation
        conductivity = base_conductivity * random.uniform(0.9, 1.1)
        conductivities.append(max(50, conductivity))
    
    return conductivities

# Generate turbidity based on system type and management
def generate_turbidity(system_list, species_list, dates):
    turbidities = []
    for i, date in enumerate(dates):
        system = system_list[i]
        species = species_list[i]
        
        # Base turbidity by system type
        if system == 'recirculating':
            base_turbidity = random.uniform(1, 8)    # Very clear
        elif system in ['tank', 'raceway']:
            base_turbidity = random.uniform(2, 15)   # Clear to slightly turbid
        elif system == 'cage':
            base_turbidity = random.uniform(1, 25)   # Natural water variation
        else:  # pond
            base_turbidity = random.uniform(5, 80)   # Most variable
        
        # Seasonal effects (higher in summer due to algae)
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        
        # Feeding-related increase in ponds
        if system == 'pond' and 10 <= date.hour <= 14:
            feeding_turbidity = random.uniform(1.2, 2.0)
        else:
            feeding_turbidity = 1.0
        
        # Weather effects (rain increases turbidity)
        weather_effect = 1.0
        if random.random() < 0.1:  # 10% chance of weather impact
            weather_effect = random.uniform(1.5, 4.0)
        
        turbidity = base_turbidity * seasonal_factor * feeding_turbidity * weather_effect
        turbidity = max(0.5, turbidity + np.random.normal(0, turbidity * 0.3))
        
        # Cap at reasonable maximum
        turbidity = min(200, turbidity)
        turbidities.append(turbidity)
    
    return turbidities

# Generate all parameters
print("Generating temperatures...")
temperatures = generate_temperature(dates, species_type, system_type)

print("Generating pH values...")
ph_values = generate_ph(dates, system_type, species_type)

print("Generating dissolved oxygen...")
do_values = generate_dissolved_oxygen(dates, temperatures, system_type, species_type)

print("Generating ammonia (NH4)...")
nh4_values = generate_ammonia_nh4(dates, system_type, species_type, temperatures)

print("Generating nitrate...")
nitrate_values = generate_nitrate(dates, nh4_values, system_type)

print("Generating conductivity...")
conductivity_values = generate_conductivity(system_type, species_type)

print("Generating turbidity...")
turbidity_values = generate_turbidity(system_type, species_type, dates)

# Create comprehensive DataFrame
data = {
    'datetime': dates,
    'farm_id': farm_ids,
    'system_type': system_type,
    'species': species_type,
    'temperature_celsius': [round(x, 1) for x in temperatures],
    'ph': [round(x, 2) for x in ph_values],
    'dissolved_oxygen_mg_per_L': [round(x, 2) for x in do_values],
    'ammonia_nh4_mg_per_L': [round(x, 4) for x in nh4_values],
    'nitrate_mg_per_L': [round(x, 2) for x in nitrate_values],
    'conductivity_uS_per_cm': [round(x, 0) for x in conductivity_values],
    'turbidity_NTU': [round(x, 2) for x in turbidity_values]
}

df = pd.DataFrame(data)
df.set_index('datetime', inplace=True)
df = df.sort_index()

# Add water quality status classifications
def classify_water_quality(row):
    """Classify overall water quality based on multiple parameters"""
    score = 0
    
    # pH check
    if 6.5 <= row['ph'] <= 8.5:
        score += 1
    elif 6.0 <= row['ph'] <= 9.0:
        score += 0.5
    
    # DO check
    species = row['species']
    min_do = 7 if species in ['trout', 'salmon'] else 5
    if row['dissolved_oxygen_mg_per_L'] >= min_do + 1:
        score += 1
    elif row['dissolved_oxygen_mg_per_L'] >= min_do:
        score += 0.5
    
    # Ammonia check
    if row['ammonia_nh4_mg_per_L'] <= 0.05:
        score += 1
    elif row['ammonia_nh4_mg_per_L'] <= 0.2:
        score += 0.5
    
    # Temperature check (simplified)
    temp = row['temperature_celsius']
    if 18 <= temp <= 28:
        score += 1
    elif 15 <= temp <= 32:
        score += 0.5
    
    # Classify based on score
    if score >= 3.5:
        return 'excellent'
    elif score >= 2.5:
        return 'good'
    elif score >= 1.5:
        return 'fair'
    else:
        return 'poor'

df['water_quality_status'] = df.apply(classify_water_quality, axis=1)

print("\n" + "="*60)
print("AQUACULTURE WATER QUALITY DATASET")
print("="*60)
print(f"Dataset size: {len(df):,} records")
print(f"Date range: {df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}")
print(f"Number of farms: {df['farm_id'].nunique()}")
print(f"System types: {', '.join(df['system_type'].unique())}")
print(f"Species: {', '.join(df['species'].unique())}")

print(f"\nWater Quality Status Distribution:")
print(df['water_quality_status'].value_counts().sort_index())

print(f"\nSystem Type Distribution:")
print(df['system_type'].value_counts())

print(f"\nSpecies Distribution:")
print(df['species'].value_counts())

print(f"\nCritical Events:")
print(f"Low oxygen events (DO <3 mg/L): {(df['dissolved_oxygen_mg_per_L'] < 3).sum():,}")
print(f"High ammonia events (NH4 >0.5 mg/L): {(df['ammonia_nh4_mg_per_L'] > 0.5).sum():,}")
print(f"pH stress events (pH <6 or >9): {((df['ph'] < 6) | (df['ph'] > 9)).sum():,}")

print(f"\nParameter Ranges (Research-Based):")
print(f"Temperature: {df['temperature_celsius'].min():.1f} - {df['temperature_celsius'].max():.1f}°C")
print(f"pH: {df['ph'].min():.2f} - {df['ph'].max():.2f}")
print(f"Dissolved Oxygen: {df['dissolved_oxygen_mg_per_L'].min():.2f} - {df['dissolved_oxygen_mg_per_L'].max():.2f} mg/L")
print(f"Ammonia (NH4): {df['ammonia_nh4_mg_per_L'].min():.4f} - {df['ammonia_nh4_mg_per_L'].max():.4f} mg/L")
print(f"Nitrate: {df['nitrate_mg_per_L'].min():.2f} - {df['nitrate_mg_per_L'].max():.2f} mg/L")
print(f"Conductivity: {df['conductivity_uS_per_cm'].min():.0f} - {df['conductivity_uS_per_cm'].max():.0f} μS/cm")
print(f"Turbidity: {df['turbidity_NTU'].min():.2f} - {df['turbidity_NTU'].max():.2f} NTU")

print(f"\nDataset Statistics:")
print(df.describe())

print(f"\nSample Data (Mixed Systems):")
sample_data = df.sample(n=8).sort_index()
print(sample_data[['farm_id', 'system_type', 'species', 'temperature_celsius', 'ph', 
                   'dissolved_oxygen_mg_per_L', 'ammonia_nh4_mg_per_L', 'water_quality_status']])

print(f"\nCorrelation Matrix:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix.round(3))

# Save to CSV
filename = 'aquaculture_water_quality_dataset.csv'
df.to_csv(filename)
print(f"\nDataset saved as '{filename}'")
print(f"File size: approximately {len(df) * len(df.columns) * 12 / 1024 / 1024:.1f} MB")

print(f"\nValidation Against Research Standards:")
print("✓ pH range (6-9): ", f"{((df['ph'] >= 6) & (df['ph'] <= 9)).mean()*100:.1f}% of samples")
print("✓ Safe NH4 (≤0.05 mg/L): ", f"{(df['ammonia_nh4_mg_per_L'] <= 0.05).mean()*100:.1f}% of samples")
print("✓ Adequate DO (≥5 mg/L): ", f"{(df['dissolved_oxygen_mg_per_L'] >= 5).mean()*100:.1f}% of samples")
print("✓ Safe nitrate (≤90 mg/L): ", f"{(df['nitrate_mg_per_L'] <= 90).mean()*100:.1f}% of samples")