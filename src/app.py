from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from test import *

app = Flask(__name__)
shark_attacks = pd.read_csv('src\\clean_GSAF5.xls.csv')

def plot_and_encode(plot_function):
    plot_function(shark_attacks)  # Pass shark_attacks to plot_function
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_encoded = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'<img src="data:image/png;base64,{plot_encoded}" alt="Plot">'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/species_attacks')
def plot_species_attacks():
    plot = plot_and_encode(speciesAttacks)
    summary = "This graph shows the distribution of shark attacks by species. It provides insights into which species are involved in more attacks. On the base data we tested on, it was found that the White Shark is the cause of the most shark attacks, causing more than double the amount of attacks than the second ranking, 4' shark. White sharks are extremely large, weighing in a range of 1500 - 2400 lbs, and can be found in all of the major oceans. White sharks are also the inspiration behind Jaws, in the famous movie."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/country_attacks')
def plot_country_attacks():
    plot = plot_and_encode(countryAttacks)
    summary = "This graph shows the distribution of shark attacks by country. It provides insights into which countries have the highest number of shark attacks. It is important to understand which countries have the most shark attacks, as emergency services in these countries can be more prepared for shark attacks if they were to ever happen. In the data we used, the United States came up as the country with the most documented shark attacks. These attacks occur on the West Coast, with the Pacific Ocean, and on the East Coast, with the Atlantic Ocean. Off the coasts, Hawaii also receives many attacks, and there are also a few attacks inside the country, with smaller sharks that reside in rivers and lakes."
    return jsonify({'plot': plot, 'summary': summary})


@app.route('/activity_attacks')
def plot_activity_attacks():
    plot = plot_and_encode(activityAttacks)
    summary = "This graph illustrates the distribution of shark attacks by activity. Understanding the activities people are engaged in when shark attacks occur is crucial for several reasons. Firstly, it allows us to identify high-risk activities and locations, helping beachgoers and authorities take appropriate precautions. Secondly, analyzing activity patterns can provide insights into human-shark interactions and factors that may trigger attacks, such as splashing or erratic movements. Moreover, knowing the activities prevalent during attacks aids in targeted educational campaigns and risk mitigation strategies. Notably, surfing emerges as one of the most prominent activities in our dataset, underscoring the importance of understanding the dynamics of this popular water sport in relation to shark encounters."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/fatal_attacks')
def plot_fatal_attacks():
    plot = plot_and_encode(fatalAttacks)
    summary = "This graph depicts the fatality rate of shark attacks, providing insights into the severity of incidents. Understanding fatal attacks is paramount for evaluating the overall risk posed by sharks and implementing effective safety measures. By analyzing factors contributing to fatal outcomes, such as location, severity of injuries, and response times, authorities can enhance emergency protocols and medical preparedness. Moreover, identifying trends in fatal attacks helps prioritize resources for areas with higher fatality rates and implement preventive measures tailored to specific circumstances. While fatal attacks represent a small percentage of overall incidents, studying them is crucial for minimizing risks and ensuring swift and appropriate responses in emergency situations."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/monthly_distribution')
def plot_monthly_distribution():
    plot = plot_and_encode(monthlyDistribution)
    summary = "This graph displays the monthly distribution of shark attacks, highlighting any seasonal patterns or trends. Understanding the monthly distribution of attacks is essential for assessing temporal variations in shark-human interactions and implementing targeted preventive measures. By identifying peak months of activity, authorities can enhance beach safety protocols, increase public awareness during high-risk periods, and allocate resources accordingly. Moreover, analyzing seasonal patterns helps researchers investigate environmental factors, such as water temperature and prey availability, that may influence shark behavior and increase the likelihood of encounters. This graph provides valuable insights into the temporal dynamics of shark attacks, aiding in the development of proactive strategies to reduce risks and ensure the safety of beachgoers year-round."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/age_distribution')
def plot_age_distribution():
    plot = plot_and_encode(ageDistribution)
    summary = "This graph illustrates the age distribution of shark attack victims, providing insights into the demographic vulnerabilities associated with such incidents. Understanding the age distribution of attacks is crucial for identifying age groups that may be at higher risk and tailoring safety measures and education campaigns accordingly. By analyzing age trends, authorities can implement targeted preventive strategies, such as age-specific beach safety guidelines and swimmer supervision programs. Moreover, studying age distribution helps researchers explore behavioral patterns and activities that may contribute to increased susceptibility to shark encounters among certain age groups. This graph sheds light on the demographic dynamics of shark attacks, facilitating the development of proactive measures to protect vulnerable populations and enhance overall beach safety."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/time_of_day_distribution')
def plot_time_of_day_distribution():
    plot = plot_and_encode(timeOfDayDistribution)
    summary = "This graph presents the distribution of shark attacks by time of day, offering insights into temporal patterns and risk factors associated with different hours. Understanding the time of day distribution of attacks is essential for assessing human activities and environmental conditions that may influence shark behavior. By analyzing peak hours of activity, authorities can implement targeted preventive measures, such as adjusting beach patrol schedules and advising beachgoers on safe swimming times. Moreover, studying time of day distribution helps researchers investigate factors like visibility and prey availability that may affect the likelihood of shark encounters during specific periods. This graph contributes valuable insights into temporal dynamics, aiding in the development of proactive strategies to mitigate risks and ensure the safety of individuals engaging in coastal activities."
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/shark_linear_regression')
def plot_shark_linear_regression():
    plot = plot_and_encode(shark_linear_regression)
    summary = ""
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/shark_naive_bias')
def plot_shark_naive_bias():
    plot = plot_and_encode(shark_naive_bias)
    summary = ""
    return jsonify({'plot': plot, 'summary': summary})

@app.route('/upload', methods=['POST'])
def upload_file():
    global shark_attacks
    if 'file' in request.files and request.files['file'].filename.endswith('.csv'):
        shark_attacks = pd.read_csv(request.files['file'])
        return jsonify({'success': True})
    elif all(key in request.form for key in ['year', 'type', 'country', 'area', 'location', 'activity', 'sex', 'age', 'injury', 'fatal', 'time', 'species', 'month', 'day']):
        new_data = {
            'Year': [request.form['year']],
            'Type': [request.form['type']],
            'Country': [request.form['country']],
            'Area': [request.form['area']],
            'Location': [request.form['location']],
            'Activity': [request.form['activity']],
            'Sex of Victim': [request.form['sex']],
            'Age': [request.form['age']],
            'Injury': [request.form['injury']],
            'Fatal (Y/N)': [request.form['fatal']],
            'Time': [request.form['time']],
            'Species': [request.form['species']],
            'Month': [request.form['month']],
            'Day - 6': [request.form['day']]
        }
        shark_attacks = shark_attacks.append(pd.DataFrame(new_data), ignore_index=True)
        shark_attacks.to_csv('src\\clean_GSAF5.xls.csv', index=False)  # save modified csv
        #reload graph when new data is added
        selected_graph = request.referrer.split("/")[-1]
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Invalid input or file format. Please provide all required fields for adding new data.'})

if __name__ == '__main__':
    app.run(debug=True)
