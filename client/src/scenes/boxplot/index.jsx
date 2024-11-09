import { Box } from "@mui/material";
import { Header, BoxPlotChart } from "../../components";

const BoxPlot = () => {
  return (
    <Box m="20px">
      <Header title="Box Plot" subtitle="Simple Box Plot" />
      <Box height="75vh">
        <BoxPlotChart />
      </Box>
    </Box>

    
  );
};

export default BoxPlot;
