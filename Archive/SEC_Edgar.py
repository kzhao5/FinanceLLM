import pandas as pd
import numpy as np
import datetime
from secfsdstools.update import update
from secfsdstools.c_index.companyindexreading import CompanyIndexReader
from secfsdstools.c_index.searching import IndexSearch
from secfsdstools.e_collector.reportcollecting import SingleReportCollector
from secfsdstools.e_filter.rawfiltering import ReportPeriodRawFilter
from secfsdstools.e_presenter.presenting import StandardStatementPresenter


#List of All Forms
FORMS_LIST = ['10-12B', '10-12G', '10-12G/A', '10-D', '10-K', '10-K/A', '10-KT', '10-KT/A', '10-Q', '10-Q/A', '10-QT', '10-QT/A', '18-K', '20-F', '20-F/A', '20FR12B', '20FR12G', '40-F', '40-F/A', '424B1', '424B2', '424B3', '424B4', '424B5', '424B7', '425', '6-K', '6-K/A', '8-K', '8-K/A', '8-K12B', '8-K12B/A', '8-K12G3', 'ARS', 'DEF 14A', 'DEF 14C', 'DEFA14A', 'DEFC14A', 'DEFM14A', 'DEFM14C', 'DEFR14A', 'F-1', 'F-1/A', 'F-3', 'F-3/A', 'F-3ASR', 'F-4', 'F-4/A', 'N-2', 'N-2/A', 'N-2ASR', 'N-2MEF', 'N-4', 'N-4/A', 'N-6/A', 'N-CSR', 'N-CSR/A', 'N-CSRS', 'N-CSRS/A', 'NT 10-Q', 'POS 8C', 'POS AM', 'POS AMI', 'POS EX', 'POSASR', 'PRE 14A', 'PREC14A', 'PREM14A', 'PRER14A', 'PRER14C', 'S-1', 'S-1/A', 'S-11', 'S-11/A', 'S-1MEF', 'S-3', 'S-3/A', 'S-3ASR', 'S-4', 'S-4/A', 'SP 15D2']
STATEMENT_LIST = ['BS', 'CF', 'CI', 'CP', 'EQ', 'IS', 'SI', 'UN']


## Company Class: Stores information from a given CIK
class Company:
    def __init__(self, cik):
        self.cik = cik
        self.report_reader = CompanyIndexReader.get_company_index_reader(cik=self.cik)
        self.consolidated_filings = pd.DataFrame(columns=['tag'])

    def get_cik(self):
        return self.cik

    def get_report_reader(self):
        return self.report_reader

    def getAvailableReports(self):
        return list(self.report_reader.get_all_company_reports_df()['form'].unique())

    def getFilingList(self, reportType, startDate, endDate):
        if reportType == 'All':
            unfilteredDF = self.report_reader.get_all_company_reports_df()
        else:
            unfilteredDF = self.report_reader.get_all_company_reports_df(forms=reportType)

        filteredDF = unfilteredDF[(unfilteredDF.period >= startDate) & (unfilteredDF.period <= endDate)]
        return filteredDF

    def appendFilings(self, df, filingDate):
        """
        Appends new filings data to the consolidated_filings DataFrame from a dataframe input.
        Includes both tag and stmt columns.

        Parameters:
        - df: pandas DataFrame containing 'tag', 'stmt', and 'merged' columns
        - filingDate: str representing the filing date to be used as column name
        """
        # Get list of tags from input DataFrame
        new_tags = df[['tag', 'stmt']].copy()

        # If this is the first data being added, initialize with both columns
        if len(self.consolidated_filings) == 0:
            self.consolidated_filings = pd.DataFrame(columns=['tag', 'stmt'])

        # Convert existing tag-stmt combinations to set for efficient comparison
        existing_combinations = set(zip(self.consolidated_filings['tag'], self.consolidated_filings['stmt']))
        new_combinations = set(zip(new_tags['tag'], new_tags['stmt']))
        combinations_to_add = new_combinations - existing_combinations

        # Add new tag-stmt combinations if any
        if combinations_to_add:
            new_rows = pd.DataFrame(list(combinations_to_add), columns=['tag', 'stmt'])
            self.consolidated_filings = pd.concat([self.consolidated_filings, new_rows], ignore_index=True)

        # Create the filing date column if it doesn't exist
        if filingDate not in self.consolidated_filings.columns:
            self.consolidated_filings[filingDate] = None

        # Update values for all tags in the input DataFrame
        for _, row in df.iterrows():
            mask = (self.consolidated_filings['tag'] == row['tag']) & (self.consolidated_filings['stmt'] == row['stmt'])
            # Update the value in the filing date column
            self.consolidated_filings.loc[mask, filingDate] = row['merged']

        # Replace all NaN with None
        self.consolidated_filings = self.consolidated_filings.replace({np.nan: None})

        return

def select_value(row):
    # Get non-NaN values and their indices
    non_nan = [(i, val) for i, val in enumerate(row) if pd.notna(val)]
    if len(non_nan) == 1:
        return non_nan[0][1]
    elif len(non_nan) == 2:
        return non_nan[-1][1]
    return None


def get_complete_filing_years(df):
    """
    Filter the dataframe to only include years with complete filing sets
    (3x 10-Q and 1x 10-K)

    Parameters:
    df (pandas.DataFrame): DataFrame containing SEC filings with columns:
        - form: Filing type (10-Q or 10-K)
        - period: Period end date

    Returns:
    pandas.DataFrame: Filtered DataFrame containing only complete filing years
    """
    # Convert period to datetime if it's not already
    df['period'] = pd.to_datetime(df['period'].astype(str), format='%Y%m%d')

    # Extract year from period
    df['filing_year'] = df['period'].dt.year

    # Create a pivot table to count filing types per year
    filing_counts = pd.pivot_table(
        df,
        index='filing_year',
        columns='form',
        values='adsh',
        aggfunc='count',
        fill_value=0
    )

    # Find years with complete sets (3x 10-Q and 1x 10-K)
    complete_years = filing_counts[
        (filing_counts['10-Q'] == 3) &
        (filing_counts['10-K'] == 1)
        ].index.tolist()

    # Filter original dataframe to only include complete years
    complete_filings = df[
        (df['filing_year'].isin(complete_years)) &
        (df['form'].isin(['10-Q', '10-K']))
        ].copy()

    # Sort by period date
    complete_filings = complete_filings.sort_values('period')

    # Drop the temporary filing_year column
    complete_filings = complete_filings.drop('filing_year', axis=1)

    return complete_filings



#Downloads complete set of 10K/Q forms
if __name__ == '__main__':
    #Update DB
    print("Updating SEC DB...")
    update()
    print("---Done.")

    #Get CIK for Each of Companies
    companyNames = [
        "Apple Inc",
        "Johnson & Johnson",
        "JPMorgan Chase",
        "Exxon",
        "Lockheed Martin",
        "NVIDIA CORP"
    ]

    #Determine Company CIK from Name
    companyObjDict = dict()
    index_search = IndexSearch.get_index_search()
    for c in companyNames:
        results = index_search.find_company_by_name(c)
        if len(results) == 1:
            print("CIK for {} : {}".format(c, results.iloc[0]['cik']))
            companyObjDict[c] = Company(cik=results.iloc[0]['cik'])
        else:
            print("-------------------------------------------------")
            print("Multiple CIK for company name {} found:".format(c))
            for index, row in results.iterrows():
                print(index, row['cik'], row['name'])
            selectedIndex = int(input("Select company index from list: "))
            companyObjDict[results.iloc[selectedIndex]['name']] = Company(cik=results.iloc[selectedIndex]['cik'])


    #Process numerical financial information using 10K/Q
    for name, obj in companyObjDict.items():
        #Get latest filings last, in order to append to np array.
        filingList = obj.getFilingList(reportType=['10-K','10-Q'], startDate=0,endDate=int(datetime.date.today().strftime('%Y%m%d'))).sort_values('period', ascending=True)
        #Determine which items of filing list have complete periods meaning three 10Q and one 10k per year
        completeFilings = get_complete_filing_years(filingList)
        print("Company {} has {} available 10K/Q reports, processing...".format(name, completeFilings.shape[0]))
        for row in completeFilings.itertuples():
            collector: SingleReportCollector = SingleReportCollector.get_report_by_adsh(adsh=row.adsh)
            rawdatabag = collector.collect()
            #Obtain data associated with this current period
            df = (rawdatabag.filter(ReportPeriodRawFilter()).join().present(StandardStatementPresenter()))
            # Obtain only the data associated from three months ended periods.
            cols_after_inpth = df.loc[:, df.columns[df.columns.get_loc('inpth') + 1:]]
            df['merged'] = cols_after_inpth.apply(select_value, axis=1)
            #Append merged column to company object after obtaining the filing date
            companyObjDict[name].appendFilings(df, row.period.strftime('%d_%m_%Y'))

    #Create datasets of completed consolidated findings.
    for name, obj in companyObjDict.items():
        filename = f"companyObjDict__{name}__consolidated_filings.csv"
        obj.consolidated_filings.to_csv(filename, index=True)


