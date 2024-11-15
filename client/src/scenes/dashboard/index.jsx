import {
  Box,
  Button,
  IconButton,
  Typography,
  useMediaQuery,
  useTheme,
  List, 
  ListItem,
  ListItemButton,
  ListItemText, 
} from "@mui/material";
import {
  Header,
  StatBox,
  LineChart,
  ProgressCircle,
  BarChart,
  GeographyChart,
} from "../../components";
import {
  DownloadOutlined,
  Email,
  PersonAdd,
  PointOfSale,
  NotInterested,
  AddReaction, 
  Traffic,
} from "@mui/icons-material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import React, {useState, useEffect, useContext} from "react";
import { api } from "../API/backend";

function Dashboard() {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const isXlDevices = useMediaQuery("(min-width: 1260px)");
  const isMdDevices = useMediaQuery("(min-width: 724px)");
  const isXsDevices = useMediaQuery("(max-width: 436px)");
  const [totalTraffic, setTraffic] = useState([]);
  const [convertedClients, setConverted] = useState([]);
  const [potentialCustomers, setPotential] = useState([]);
  const [conversionRate, setRate] = useState([]);
  const [campaignReach, setCampaign] = useState([]);
  const [latestEngage, setLatest] = useState([]);
  const [adSpend, setSpend] = useState([]); 
  const [predROI, setROI] = useState([]);  
  const [recoBySeg, setReco] = useState([]);  

  useEffect(() => {
    api.get('recoBySeg')
      .then((res) => res.data.data)
      .then((data) => {
        setReco(data);
    });
  }, []);

  useEffect(() => {
    api.get('totalTraffic')
      .then((res) => res.data.data)
      .then((data) => {
        setTraffic(data);
    });
  }, []);
  // console.log(totalTraffic)

  useEffect(() => {
    api.get('convertedClients')
      .then((res) => res.data.data)
      .then((data) => {
        setConverted(data);
    });
  }, []);

  useEffect(() => {
    api.get('potentialCustomers')
      .then((res) => res.data.data)
      .then((data) => {
        setPotential(data);
    });
  }, []);

  useEffect(() => {
    api.get('conversionRate')
      .then((res) => res.data.data)
      .then((data) => {
        setRate(data);
    });
  }, []);

  useEffect(() => {
    api.get('campaignReach')
      .then((res) => res.data.data)
      .then((data) => {
      setCampaign(data);
    });
  }, []);
  
  useEffect(() => {
    api.get('latestEngage')
      .then((res) => res.data.data)
      .then((data) => {
      setLatest(data);
    });
  }, []);

  useEffect(() => {
    api.get('adSpend')
      .then((res) => res.data.data)
      .then((data) => {
        setSpend(data);
    });
  }, []);

  useEffect(() => {
    api.get('predROI')
      .then((res) => res.data.data)
      .then((data) => {
        setROI(data);
    });
  }, []);

  // console.log(predROI)
  
  const products = [{
    id: 0,
    item: 'Fixed Deposits',
  }, {
    id: 1,
    item: 'Credit & Debit Card',
  }, {
    id: 2,
    item: 'Account',
  }, {
    id: 3,
    item: 'Loan',  
  }];


  return (
    <Box m="20px">
      <Box display="flex" justifyContent="space-between">
        <Header title="Campaign Tracking" subtitle="Welcome to your dashboard" />
        {!isXsDevices && (
          <Box>
            <Button
              variant="contained"
              sx={{
                bgcolor: colors.blueAccent[700],
                color: "#fcfcfc",
                fontSize: isMdDevices ? "14px" : "10px",
                fontWeight: "bold",
                p: "10px 20px",
                mt: "18px",
                transition: ".3s ease",
                ":hover": {
                  bgcolor: colors.blueAccent[800],
                },
              }}
              startIcon={<DownloadOutlined />}
            >
              DOWNLOAD REPORTS
            </Button>
          </Box>
        )}
      </Box>

      {/* GRID & CHARTS */}
      <Box
        display="grid"
        gridTemplateColumns={
          isXlDevices
            ? "repeat(12, 1fr)"
            : isMdDevices
            ? "repeat(6, 1fr)"
            : "repeat(3, 1fr)"
        }
        gridAutoRows="140px"
        gap="20px"
      >
        {/* Statistic Items */}
        <Box
          gridColumn="span 3"
          bgcolor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title={ totalTraffic }
            subtitle="Traffic Received"
            // progress="0.19036168720569081"
            // increase="+14%"
            icon={
              <Traffic
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title={ convertedClients }
            subtitle="Converted Clients"
            // progress="0.50"
            // increase="+21%"
            icon={
              <AddReaction
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title={ potentialCustomers }
            subtitle="Potential Clients"
            // progress="0.30"
            // increase="+5%"
            icon={
              <PersonAdd
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title={ conversionRate }
            subtitle="Click Through Rate"
            // progress="0.80"
            // increase="+43%"
            icon={
              <Traffic
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>

        {/* ---------------- Row 2 ---------------- */}

        {/* Line Chart */}
        <Box
          gridColumn={
            isXlDevices ? "span 8" : isMdDevices ? "span 6" : "span 4"
          }
          gridRow="span 2"
          bgcolor={colors.primary[400]}
        >
          <Box
            mt="25px"
            px="30px"
            display="flex"
            justifyContent="space-between"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.gray[100]}
              >
                Campaign Reach
              </Typography>
              {/* <Typography
                variant="h5"
                fontWeight="bold"
                color={colors.greenAccent[500]}
              >
                $59,342.32
              </Typography> */}
            </Box>
            <IconButton>
              <DownloadOutlined
                sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
              />
            </IconButton>
          </Box>
          <Box height="250px" mt="-20px">
            <BarChart 
            data = { campaignReach }
            index = {"date"}
            keys =  {['scrolled', 'clicked', 'credentials', 'converted',]}
            isDashboard={false} />
          </Box>
        </Box>
        

        {/* Transaction Data */}
        <Box
          gridColumn={isXlDevices ? "span 4" : "span 3"}
          gridRow="span 2"
          bgcolor={colors.primary[400]}
          overflow="auto"
        >
          <Box borderBottom={`4px solid ${colors.primary[500]}`} p="15px" >
            <Typography color={colors.gray[100]} variant="h5" fontWeight="600" justify="space-between">
              Recent Engagements
            </Typography>
          </Box>

          {latestEngage.map((transaction, index) => (
            <Box
              key={`${transaction.txId}-${index}`}
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              borderBottom={`4px solid ${colors.primary[500]}`}
              p="15px"
            >
              <Box>
                <Typography
                  color={colors.greenAccent[500]}
                  variant="h5"
                  fontWeight="600"
                >
                  {transaction.action}
                </Typography>
                <Typography color={colors.gray[100]}>
                  {transaction.id}
                </Typography>
              </Box>
              <Typography color={colors.gray[100]}>
                {transaction.date}
              </Typography>
              <Typography color={colors.gray[100]}>
                {transaction.score}/5 Rating
              </Typography>
              {/* <Box
                bgcolor={colors.greenAccent[500]}
                p="5px 10px"
                borderRadius="4px"
              >
                ${transaction.cost}
              </Box> */}
            </Box>
          ))}
        </Box>

        {/* Line Chart */}
        <Box
          gridColumn={
            isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
          }
          gridRow="span 2"
          bgcolor={colors.primary[400]}
        >
        <Box
            mt="25px"
            px="30px"
            display="flex"
            justifyContent="space-between"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.gray[100]}
              >
                Ad Spending
              </Typography>
              {/* <Typography
                variant="h5"
                fontWeight="bold"
                color={colors.greenAccent[500]}
              >
                $59,342.32
              </Typography> */}
            </Box>
            <IconButton>
              <DownloadOutlined
                sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
              />
            </IconButton>
          </Box>
          <Box height="250px" mt="-20px">
            <BarChart 
            data = { adSpend }
            index = {"date"}
            keys =  {['spending']}
            isDashboard={false} />
          </Box>
        
        </Box>

        {/* Line Chart */}
        <Box
          gridColumn={
            isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
          }
          gridRow="span 2"
          bgcolor={colors.primary[400]}
        >
          <Box
            mt="25px"
            px="30px"
            display="flex"
            justifyContent="space-between"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.gray[100]}
              >
                Predicted ROI 
              </Typography>
              {/* <Typography
                variant="h5"
                fontWeight="bold"
                color={colors.greenAccent[500]}
              >
                $59,342.32
              </Typography> */}
            </Box>
            <IconButton>
              <DownloadOutlined
                sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
              />
            </IconButton>
          </Box>
          <Box height="250px" mt="-20px">
            <BarChart 
            data = { predROI }
            index = {"Campaign Month"}
            keys =  {['clicks', 'leads', 'orders']}
            isDashboard={false} />
          </Box>
        
        </Box>  
        <Box
          gridColumn={
            isXlDevices ? "span 12" : isMdDevices ? "span 6" : "span 4"
          }
          gridRow="span 2"
          bgcolor={colors.primary[400]}
        >
          <Box
            mt="25px"
            px="30px"
            display="flex"
            justifyContent="space-between"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.gray[100]}
              >
                Recommendations
              </Typography>
            </Box>
            <IconButton>
              <DownloadOutlined
                sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
              />
            </IconButton>
          </Box>
          <Box height="250px" mt="-20px">
            <BarChart 
            data = { recoBySeg }
            index = {"segment"}
            keys =  {["deposits_reco", "cards_reco", "account_reco", "loan_reco"]}
            isDashboard={false} />
          </Box>
        
        </Box>
         
          
        
      </Box>
    </Box>
  );
}

export default Dashboard;
