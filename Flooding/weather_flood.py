import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

import datetime
from datetime import timedelta

import seaborn as sns
import matplotlib.pyplot as plt


class flooding:
    '''
    Performs different analysis related flooding
    '''

    def load_data(self):
        '''
        Loads data from the file
        '''
        xls = pd.ExcelFile("C:\\Users\\mohammad.sharif\\Desktop\\DS_models\\Flooding\\SevereWeather.xlsx")
        self.df = pd.read_excel(xls, 'Raw Severe Data')
        self.distinct_year = self.df.Year.unique()
        self.distinct_EVENT_TYPE = self.df.EVENT_TYPE.unique()

        # df1 = self.df['EVENT_TYPE'].unique()
        # print(np.sort(df1))

    def selected_data(self, interesting_county, interesting_district):
        '''
        Slices relevant data as needed
        '''

        #self.df = self.df.loc[self.df['CORRECTED_NAME'].isin(interesting_county)]
        self.df = self.df.loc[self.df['DistrictName'].isin(interesting_district)]

    def find_event_seq(self, interesting_events, date_difference):
        '''
        Returns events preceding to any flooding events within certain days window
        '''

        f = 0
        p = 0
        df_casual = pd.DataFrame()
        df = self.df
        df = df.sort_values(by='BEGIN_DATE_TIME', ascending=True)
        df.index = range(len(df.index))
        for i in range(1, len(df)):
            if (df.loc[i, 'EVENT_TYPE']) in interesting_events:
                j = 1
                diff = df.loc[i, 'BEGIN_DATE_TIME'] - df.loc[i - j, 'BEGIN_DATE_TIME']

                while (diff.days < date_difference):
                    if ((df.loc[i, 'CORRECTED_NAME']) == (df.loc[i - j, 'CORRECTED_NAME'])) & (
                            df.loc[i - j, 'EVENT_TYPE'] not in interesting_events):
                        df_casual.loc[p, 'start_time_1st_event'] = df.loc[i - j, 'BEGIN_DATE_TIME']
                        df_casual.loc[p, 'start_time_2nd_event'] = df.loc[i, 'BEGIN_DATE_TIME']
                        df_casual.loc[p, '1st_event'] = df.loc[i - j, 'EVENT_TYPE']
                        df_casual.loc[p, '2nd_event'] = df.loc[i, 'EVENT_TYPE']
                        df_casual.loc[p, "CORRECTED_NAME"] = df.loc[i, 'CORRECTED_NAME']
                        f = f + 1

                        p = p + 1
                    j = j + 1
                    if (i - j < 0): break
                    diff = df.loc[i, 'BEGIN_DATE_TIME'] - df.loc[i - j, 'BEGIN_DATE_TIME']
            # if (f>100): break

        df_casual = df_casual.reset_index(drop=True)
        print(df_casual)
        df_casual.to_csv('C:\\Users\\mohammad.sharif\\Desktop\\DS_models\\Flooding\\causing_event.csv')
        return (df_casual)

    def find_flood_difference(self, interesting_events):
        '''
        Returns yearly flooding event count changes
        '''

        df = self.df
        df = df.loc[df['EVENT_TYPE'].isin(interesting_events)]

        # df['year'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).year
        df['Month'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).month
        df['Day'] = pd.DatetimeIndex(df['BEGIN_DATE_TIME']).day

        # df = df.groupby(['CZ_NAME','year']).size().reset_index(name='counts')
        df = df.groupby(['CORRECTED_NAME', 'Year']).size().reset_index(name='counts')
        # mean = df.groupby(['CZ_NAME','year']).agg(Mean=('counts', 'mean'))
        # mean = df.groupby(['CORRECTED_NAME','year']).agg(Mean=('counts', 'mean'))
        # std = df.groupby(['CORRECTED_NAME','year']).agg(Std=('counts', 'std'))

        df = df.sort_values(by=['CORRECTED_NAME', 'Year'], ascending=True)
        # print(df)

        demean = lambda x: (x - x.mean()) / np.std(x)
        counts_normalized = df.groupby(['CORRECTED_NAME']).transform(demean)["counts"]

        df['counts_normalized'] = counts_normalized
        print(df.loc[df["CORRECTED_NAME"] == 'RICHMOND'])
        return (df)

    def visualize_hitmap(self, df_vz):
        '''
        Draws hitmap visualization of yearly flooding changes by county and year
        '''

        df = df_vz

        # define the plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Add title to the Heat map
        title = "Flooding Changes in Consecutive Years (normalised deviation))"

        # Set the font size and the distance of the title
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])

        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        df['counts_normalized_diff'] = df.counts_normalized.diff() + 3

        # Draw hitmap
        heatmap2_data = pd.pivot_table(df, values='counts_normalized_diff', index=['CORRECTED_NAME'], columns='Year')
        sns.heatmap(heatmap2_data, cmap="BuGn")
        plt.show()

        return

    def flooding_test(self, df):
        '''
        Tets Flooding situtation
        '''

        year = datetime.datetime.today().year - 1

        for year in list(range(year, year - 20, -1)):
            if year not in df['Year'].values:
                df = df.append({'Year': year, 'counts': 0}, ignore_index=True)


        df = df.sort_values(by='Year', ascending=True)
        #print(df['counts'][:10])
        print(df)

        stat, p = ttest_ind(df['counts'][:10], df['counts'][10:], equal_var=False)
        p = p / 2  # convert to one tail from two tail
        #print('Statistics=%.3f, p=%.3f' % (stat, p))
        print('p=%.3f' % (p))

        # interpret
        alpha = 0.05
        if p > alpha:
            print('Same distributions (fail to reject H0)')
        else:
            print('Different distributions (reject H0)')


    def plot_flood_vs_other(self, interesting_county, interesting_district):
        '''
        Draws scatter plots for different events with flooding for comparison
        '''

        df = self.df
        #interesting_county = ['STAFFORD']


        #df = df.loc[df['CORRECTED_NAME'].isin(interesting_county)]
        df = df.loc[df['DistrictName'].isin(interesting_district)]

        # distinct_county = df.CORRECTED_NAME.unique()
        #self.distinct_year = df.Year.unique()
        #self.distinct_EVENT_TYPE = df.EVENT_TYPE.unique()


        #df = df.groupby(['CORRECTED_NAME', 'Year', 'EVENT_TYPE'])['EPISODE_ID'].nunique().reset_index(name='counts')
        #df = df.groupby(['Year', 'EVENT_TYPE'])['EPISODE_ID'].nunique().reset_index(name='counts')
        df = df.groupby(['Year', 'EVENT_TYPE']).size().reset_index(name='counts')

        #county = 'STAFFORD'


        tuple = []
        for year in self.distinct_year:
            for event in self.distinct_EVENT_TYPE:
                list = [year, event]
                tuple.append(list)

        grouped_set = []
        for index, row in df.iterrows():
            grouped_set.append([row['Year'], row['EVENT_TYPE']])
        # print(grouped_set)

        for element in tuple:
            if element not in grouped_set:
                #df = df.append({'CORRECTED_NAME': county, 'Year': element[0], 'EVENT_TYPE': element[1], 'counts': 0}, ignore_index=True)
                df = df.append({'Year': element[0], 'EVENT_TYPE': element[1], 'counts': 0},
                               ignore_index=True)


        #df = self.add_new_events_for_visualization(df,'EVENT_TYPE')
        year = np.asarray(sorted(df.Year.unique()))
        print(year)
        Dense_Fog = np.asarray(df[df['EVENT_TYPE'] == 'Dense Fog'].sort_values(['Year']).counts)
        print(Dense_Fog)
        Heavy_Rain = np.asarray(df[df['EVENT_TYPE'] == 'Heavy Rain'].sort_values(['Year']).counts)
        print(Heavy_Rain)
        Heavy_Snow = np.asarray(df[df['EVENT_TYPE'] == 'Heavy Snow'].sort_values(['Year']).counts)
        print(Heavy_Snow)
        High_Wind = np.asarray(df[df['EVENT_TYPE'] == 'High Wind'].sort_values(['Year']).counts)
        print(High_Wind)
        # Hurricane_Typhoon = np.asarray(df[df['EVENT_TYPE'] == 'Hurricane(Typhoon)'].sort_values(['Year']).counts)
        # print(Hurricane_Typhoon)
        Ice_Storm = np.asarray(df[df['EVENT_TYPE'] == 'Ice Storm'].sort_values(['Year']).counts)
        print(Ice_Storm)
        Winter_Storm = np.asarray(df[df['EVENT_TYPE'] == 'Winter Storm'].sort_values(['Year']).counts)
        print(Winter_Storm)
        Flash_Flood = np.asarray(df[df['EVENT_TYPE'] == 'Flash Flood'].sort_values(['Year']).counts)
        print(Flash_Flood)
        Flood = np.asarray(df[df['EVENT_TYPE'] == 'Flood'].sort_values(['Year']).counts)
        print(Flood)


        df = pd.DataFrame({'Year': year, 'Dense_Fog': Dense_Fog, 'Heavy_Rain': Heavy_Rain, 'Heavy_Snow': Heavy_Snow,
                           'High_Wind': High_Wind, 'Ice Storm': Ice_Storm, 'Winter_Storm': Winter_Storm,
                           'Flash_Flood': Flash_Flood, 'Flood': Flood})

        title = 'Yearly Event Counts Comparison (Fredericksburg)'
        ncol = 8
        self.plot_yearly_events(df, ncol, title)
        # plot aggregated events count comparison with flooding
        #flooding = np.add(Flood, Flash_Flood)
        flooding = Flood
        add_array = np.array([Dense_Fog, Heavy_Rain,  Heavy_Snow, High_Wind, Ice_Storm,  Winter_Storm])
        others = add_array.sum(axis=0)
        agg_df = pd.DataFrame({'Year': year,'others': others, 'flooding': flooding })
        title = 'Flooding vs other event(aggregated) count comparison  (Fredericksburg)'
        ncol = 2
        self.plot_yearly_events(agg_df, ncol, title)

    def plot_event_seq(self, event_seq_df, interesting_county, interesting_district):

        event_seq_df = pd.read_csv('C:\\Users\\mohammad.sharif\\Desktop\\DS_models\\Flooding\\causing_event.csv')

        event_seq_df = event_seq_df.loc[event_seq_df['CORRECTED_NAME'].isin(interesting_county)]
        #event_seq_df = event_seq_df.loc[event_seq_df['DistrictName'].isin(interesting_district)]
        event_seq_df['Year'] = pd.DatetimeIndex(event_seq_df['start_time_1st_event']).year
        event_seq_df = event_seq_df.groupby(['Year', '1st_event']).size().reset_index(name='counts')


        tuple = []
        for year in self.distinct_year:
            for event in self.distinct_EVENT_TYPE:
                list = [year, event]
                tuple.append(list)

        grouped_set = []
        for index, row in event_seq_df.iterrows():
            grouped_set.append([row['Year'], row['1st_event']])
        # print(grouped_set)

        for element in tuple:
            if element not in grouped_set:
                # df = df.append({'CORRECTED_NAME': county, 'Year': element[0], 'EVENT_TYPE': element[1], 'counts': 0}, ignore_index=True)
                event_seq_df = event_seq_df.append({'Year': element[0], '1st_event': element[1], 'counts': 0},
                               ignore_index=True)



        #event_seq_df = self.add_new_events_for_visualization(event_seq_df, '1st_event')
        year = np.asarray(sorted(event_seq_df.Year.unique()))
        print(year)
        Dense_Fog = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Dense Fog'].sort_values(['Year']).counts)
        print(Dense_Fog)

        Hail = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Hail'].sort_values(['Year']).counts)
        print(Hail)

        Heavy_Rain = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Heavy Rain'].sort_values(['Year']).counts)
        print(Heavy_Rain)

        High_Wind = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'High Wind'].sort_values(['Year']).counts)
        print(High_Wind)

        Lightning = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Lightning'].sort_values(['Year']).counts)
        print(Lightning)

        Thunderstorm_Wind = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Thunderstorm Wind'].sort_values(['Year']).counts)
        print(Thunderstorm_Wind)

        Tornado = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Tornado'].sort_values(['Year']).counts)
        print(Tornado)

        Tropical_Storm = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Tropical Storm'].sort_values(['Year']).counts)
        print(Tropical_Storm)

        Winter_Weather = np.asarray(event_seq_df[event_seq_df['1st_event'] == 'Winter Weather'].sort_values(['Year']).counts)
        print(Winter_Weather)

        df = pd.DataFrame({'Year': year, 'Dense_Fog': Dense_Fog, 'Hail': Hail, 'Heavy_Rain': Heavy_Rain,
                           'High_Wind': High_Wind, 'Lightning': Lightning, 'Thunderstorm_Wind': Thunderstorm_Wind,
                           'Tornado': Tornado, 'Tropical_Storm': Tropical_Storm, 'Winter_Weather':Winter_Weather})

        title = 'Yearly prior event counts within 7 days before flooding (Fredericksburg)'
        ncol = 9
        self.plot_yearly_events(df, ncol, title)

    def plot_yearly_events(self, df, ncol, title):
        '''
                #draws scatter plot for different criteria
        '''

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                  "#17becf"]
        i = 0
        # fig, ax = plt.subplots(1, 1)
        fig, axs = plt.subplots(ncol, sharey=True, squeeze=False)

        fig.suptitle(title)
        for column in df.drop('Year', axis=1):
            # print(df['Year'])
            axs[i][0].scatter(df['Year'], df[column], color=colors[i], label=column, s=df[column] * 20)
            # axs[i][0].plot(df['Year'], df[column], color=colors[i], label=column)
            # axs[i][0].legend(loc='center left', bbox_to_anchor=(0.82, 1), fancybox=True, shadow=True, ncol=3)
            axs[i][0].legend(loc='center right', bbox_to_anchor=(-0.03, 0.5), fancybox=True, shadow=True, markerscale=0)
            axs[i][0].figure.set_size_inches(15, 3)
            #axs[i][0].set_ylim(bottom=1)
            axs[i][0].set_xticks(df['Year'])
            axs[i][0].grid(True, linestyle='--', linewidth=0.3)
            if (i < ncol - 1):
                axs[i][0].set_xticklabels([])

            i = i + 1

        plt.show()


    def add_new_events_for_visualization(self, event_seq_df, event):

        tuple = []
        for year in self.distinct_year:
            for event in self.distinct_EVENT_TYPE:
                list = [year, event]
                tuple.append(list)

        grouped_set = []
        for index, row in event_seq_df.iterrows():
            #grouped_set.append([row['Year'], row['1st_event']])
            grouped_set.append([row['Year'], row[event]])
        # print(grouped_set)

        for element in tuple:
            if element not in grouped_set:
                # df = df.append({'CORRECTED_NAME': county, 'Year': element[0], 'EVENT_TYPE': element[1], 'counts': 0}, ignore_index=True)
                event_seq_df = event_seq_df.append({'Year': element[0], '1st_event': element[1], 'counts': 0},
                               ignore_index=True)

        return event_seq_df


    def group_items(self, groupby_items, interesting_events, interesting_county, interesting_district):
        df = self.df
        #df = df.loc[df['CORRECTED_NAME'].isin(interesting_county)]
        df = df.loc[df['DistrictName'].isin(interesting_district)]
        df = df.loc[df['EVENT_TYPE'].isin(interesting_events)]

        df = df.groupby(groupby_items).size().reset_index(name='counts')
        #df = df.groupby(['CORRECTED_NAME', 'Year']).size().reset_index(name='counts')

        # mean = df.groupby(['CZ_NAME','year']).agg(Mean=('counts', 'mean'))
        # mean = df.groupby(['CORRECTED_NAME','year']).agg(Mean=('counts', 'mean'))
        # std = df.groupby(['CORRECTED_NAME','year']).agg(Std=('counts', 'std'))

        df = df.sort_values(by=groupby_items, ascending=True)
        #df = df.sort_values(by=['CORRECTED_NAME', 'Year'], ascending=True)
        return df



def main():
    interesting_events = ['Flood', 'Flash Flood']
    interesting_district = ['Fredericksburg']
    #interesting_district = ['Bristol','Culpeper','Fredericksburg','Hampton Roads','Lynchburg', 'Northern Virginia', 'Richmond', 'Salem', 'Staunton']
    interesting_county = ['CAROLINE', 'ESSEX', 'FREDERICKSBURG', 'GLOUCESTER', 'KING GEORGE', 'KING AND QUEEN',
                          'KING WILLIAM', 'LANCASTER', 'MATHEWS', 'MIDDLESEX', 'NORTHUMBERLAND', 'SPOTSYLVANIA',
                          'STAFFORD', 'RICHMOND', 'WESTMORELAND']

    #groupby_items = ['CORRECTED_NAME', 'Year']
    groupby_items = ['Year']

    date_duration = 7
    flood = flooding()
    flood.load_data()
    # flood.selected_data(interesting_county)
    #event_seq_df = flood.find_event_seq(interesting_events, date_duration)
    # event_diff_df = flood.find_flood_difference(interesting_events)
    # flood.visualize_hitmap(event_diff_df)
    #interesting_events = ['Flood', 'Flash Flood']
    interesting_events = ['Flood']

    df = flood.group_items(groupby_items, interesting_events, interesting_county, interesting_district)
    #print(df)
    flood.flooding_test(df)
    #flood.plot_flood_vs_other(interesting_county, interesting_district)
    #flood.plot_event_seq(event_seq_df="", interesting_county=interesting_county, interesting_district = interesting_district)


if __name__ == '__main__':
    main()
